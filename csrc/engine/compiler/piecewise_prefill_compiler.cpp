#include "piecewise_prefill_compiler.hpp"

#include "../../global_state/ar_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../global_state/hang_trace.hpp"
#include "../compiled_prefill_flags.hpp"
#include "../../utils.hpp"
#include "../../utils/agent_debug.hpp"
#include "piecewise_bucket_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <spdlog/spdlog.h>

namespace infinilm::engine {

namespace {

bool rank_worker_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_RANK_WORKER_PROFILE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// Trim redundant post-allreduce barrier (next layer's pre-graph barrier subsumes it).
/// Default on; opt out for bisect: INFINI_PIECEWISE_KEEP_BARRIERS=1
bool piecewise_trim_barriers() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_PIECEWISE_KEEP_BARRIERS");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 0 : 1;
    }
    return cached == 1;
}

double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

size_t compute_prefill_len(const InfinilmModel::Input &input) {
    if (input.input_offsets.has_value()) {
        const auto &offsets = input.input_offsets.value();
        const size_t n = offsets->size(0);
        if (n >= 2) {
            auto cpu_offsets = offsets->to(infinicore::Device::cpu());
            const auto *data = reinterpret_cast<const int32_t *>(cpu_offsets->data());
            return static_cast<size_t>(data[n - 1] - data[0]);
        }
    }
    if (input.input_ids.has_value()) {
        return input.input_ids.value()->size(1);
    }
    return 0;
}

void set_attn_metadata(const InfinilmModel::Input &input) {
    infinilm::global_state::get_forward_context().attn_metadata = {
        input.past_sequence_lengths,
        input.total_sequence_lengths,
        input.input_offsets,
        input.cu_seqlens,
        input.block_tables,
        input.slot_mapping,
    };
}

void set_attn_metadata_for_varlen_batch(const InfinilmModel::Input &compiled,
                                        const InfinilmModel::Input &runtime) {
    const size_t runtime_n_req = runtime.block_tables.value()->size(0);
    const size_t block_per_req = runtime.block_tables.value()->size(1);
    const size_t offset_len = runtime.input_offsets.value()->size(0);
    const size_t cu_len = runtime.cu_seqlens.value()->size(0);
    const size_t slot_len = runtime.slot_mapping.value()->shape()[0];

    auto &meta = infinilm::global_state::get_forward_context().attn_metadata;
    meta.past_sequence_lengths = compiled.past_sequence_lengths.has_value()
                                     ? std::optional<infinicore::Tensor>(
                                           compiled.past_sequence_lengths.value()->narrow({{0, 0, runtime_n_req}}))
                                     : std::nullopt;
    meta.total_sequence_lengths = compiled.total_sequence_lengths.value()->narrow({{0, 0, runtime_n_req}});
    meta.input_offsets = compiled.input_offsets.value()->narrow({{0, 0, offset_len}});
    meta.cu_seqlens = compiled.cu_seqlens.value()->narrow({{0, 0, cu_len}});
    meta.block_tables = compiled.block_tables.value()->narrow({{0, 0, runtime_n_req}, {1, 0, block_per_req}});
    // paged_caching uses slot_mapping.shape[0] as num_tokens (see paged_caching/info.h).
    meta.slot_mapping = compiled.slot_mapping.value()->narrow({{0, 0, slot_len}});

    // #region agent log
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        const int32_t rt_cu0 = infinilm::agent_debug::first_int32(runtime.cu_seqlens.value());
        const int32_t rt_cuN = infinilm::agent_debug::last_int32(runtime.cu_seqlens.value());
        const int32_t cp_cu0 = infinilm::agent_debug::first_int32(compiled.cu_seqlens.value()->narrow({{0, 0, cu_len}}));
        const int32_t cp_cuN = infinilm::agent_debug::last_int32(compiled.cu_seqlens.value()->narrow({{0, 0, cu_len}}));
        const int32_t meta_cu0 = infinilm::agent_debug::first_int32(meta.cu_seqlens.value());
        const int32_t meta_cuN = infinilm::agent_debug::last_int32(meta.cu_seqlens.value());
        infinilm::agent_debug::log(
            "piecewise_prefill_compiler.cpp:set_attn_metadata_for_varlen_batch",
            "pw_attn_meta",
            "W1",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"slot_len\":" +
                std::to_string(slot_len) + ",\"rt_cu0\":" + std::to_string(rt_cu0) +
                ",\"rt_cuN\":" + std::to_string(rt_cuN) + ",\"cp_cu0\":" + std::to_string(cp_cu0) +
                ",\"cp_cuN\":" + std::to_string(cp_cuN) + ",\"meta_cu0\":" + std::to_string(meta_cu0) +
                ",\"meta_cuN\":" + std::to_string(meta_cuN) + "}",
            "meta-check");
    }
    // #endregion
}

void zero_tensor_tail_seq_(infinicore::Tensor &tensor, size_t valid_len, size_t bucket) {
    if (valid_len >= bucket) {
        return;
    }
    auto tail = tensor->narrow({{1, valid_len, bucket - valid_len}});
    set_zeros(tail);
}

void clear_stale_bucket_tails_(global_state::PiecewisePrefillState &piecewise,
                               infinicore::Tensor &logits_holder,
                               size_t valid_seq_len,
                               size_t bucket) {
    if (valid_seq_len >= bucket) {
        return;
    }
    zero_tensor_tail_seq_(piecewise.hidden_states, valid_seq_len, bucket);
    zero_tensor_tail_seq_(piecewise.residual, valid_seq_len, bucket);
    for (auto &staging : piecewise.layer_staging) {
        zero_tensor_tail_seq_(staging.q_rope, valid_seq_len, bucket);
        zero_tensor_tail_seq_(staging.k_rope, valid_seq_len, bucket);
        zero_tensor_tail_seq_(staging.v_rope, valid_seq_len, bucket);
        zero_tensor_tail_seq_(staging.attn_output, valid_seq_len, bucket);
    }
    if (piecewise.ar_staging) {
        zero_tensor_tail_seq_(piecewise.ar_staging, valid_seq_len, bucket);
    }
    if (logits_holder) {
        zero_tensor_tail_seq_(logits_holder, valid_seq_len, bucket);
    }
}

} // namespace

PiecewisePrefillCompiler::PiecewisePrefillCompiler(const std::shared_ptr<InfinilmModel> &model,
                                                   RankBarrier *barrier)
    : model_(model), barrier_(barrier) {
    enabled_ = native_piecewise_prefill_enabled() && model_->supports_native_piecewise_prefill();
    if (!enabled_) {
        return;
    }
    max_seq_len_ = compile_max_seq_from_env();
    auto pad_ladder = piecewise_compile_buckets(max_seq_len_);
    bs_to_padded_ = build_bs_to_padded_bucket(pad_ladder);
    capture_buckets_ = piecewise_capture_buckets(max_seq_len_);
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_CAPTURE_BUCKETS")) {
        capture_buckets_.clear();
        std::string spec(raw);
        size_t start = 0;
        while (start < spec.size()) {
            const size_t comma = spec.find(',', start);
            const std::string token = spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!token.empty()) {
                capture_buckets_.push_back(static_cast<size_t>(std::stoul(token)));
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        // Pad ladder must match capture buckets when overridden (e.g. smoke bucket=256).
        bs_to_padded_ = build_bs_to_padded_bucket(capture_buckets_);
    }
    std::sort(capture_buckets_.begin(), capture_buckets_.end(), std::greater<size_t>());
}

void PiecewisePrefillCompiler::allocate_layer_staging_(size_t bucket, size_t num_layers) {
    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.bucket_seq_len = bucket;
    piecewise.layer_staging.clear();
    piecewise.layer_staging.resize(num_layers);
    const auto device = infinicore::context::getDevice();
    const auto &model_config = infinilm::global_state::get_infinilm_config().model_config;
    const auto dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t tp_size = std::max<size_t>(
        1, static_cast<size_t>(infinilm::global_state::get_tensor_model_parallel_world_size()));
    const size_t num_heads = model_config->get<size_t>("num_attention_heads") / static_cast<size_t>(tp_size);
    const size_t total_kv = model_config->get<size_t>("num_key_value_heads");
    const size_t num_kv_heads = total_kv < tp_size ? 1 : total_kv / tp_size;
    const size_t head_dim = model_config->get_head_dim();

    for (size_t i = 0; i < num_layers; ++i) {
        auto &st = piecewise.layer_staging[i];
        st.q_rope = infinicore::Tensor::empty({1, bucket, num_heads, head_dim}, dtype, device);
        st.k_rope = infinicore::Tensor::empty({1, bucket, num_kv_heads, head_dim}, dtype, device);
        st.v_rope = infinicore::Tensor::empty({1, bucket, num_kv_heads, head_dim}, dtype, device);
        st.attn_output = infinicore::Tensor::empty({1, bucket, num_heads * head_dim}, dtype, device);
    }
    piecewise.hidden_states = infinicore::Tensor::empty({1, bucket, hidden}, dtype, device);
    piecewise.residual = infinicore::Tensor::empty({1, bucket, hidden}, dtype, device);
    piecewise.ar_staging = infinicore::Tensor::empty({1, bucket, hidden}, dtype, device);
}

InfinilmModel::Input PiecewisePrefillCompiler::make_bucket_input_(size_t bucket, size_t nblocks, size_t n_req) const {
    InfinilmModel::Input input;
    const auto device = infinicore::context::getDevice();
    input.input_ids = infinicore::Tensor::empty({1, bucket}, infinicore::DataType::I64, device);
    input.position_ids = infinicore::Tensor::empty({bucket}, infinicore::DataType::I64, device);
    input.past_sequence_lengths = infinicore::Tensor::empty({n_req}, infinicore::DataType::I32, device);
    input.total_sequence_lengths = infinicore::Tensor::empty({n_req}, infinicore::DataType::I32, device);
    set_zeros(input.input_ids.value());
    set_zeros(input.past_sequence_lengths.value());
    set_zeros(input.total_sequence_lengths.value());

    std::vector<int64_t> position_ids_vec(bucket);
    std::iota(position_ids_vec.begin(), position_ids_vec.end(), int64_t{0});
    infinicore::context::memcpyH2D(
        input.position_ids.value()->data(), position_ids_vec.data(), bucket * sizeof(int64_t), false);

    std::vector<int32_t> past_lengths_vec(n_req, 0);
    std::vector<int32_t> total_lengths_vec(n_req, static_cast<int32_t>(bucket / std::max<size_t>(1, n_req)));
    infinicore::context::memcpyH2D(
        input.past_sequence_lengths.value()->data(),
        past_lengths_vec.data(),
        n_req * sizeof(int32_t),
        false);
    infinicore::context::memcpyH2D(
        input.total_sequence_lengths.value()->data(),
        total_lengths_vec.data(),
        n_req * sizeof(int32_t),
        false);

    input.input_offsets = infinicore::Tensor::empty({n_req + 1}, infinicore::DataType::I32, device);
    std::vector<int32_t> input_offsets_vec(n_req + 1, 0);
    const int32_t per_req = static_cast<int32_t>(bucket / std::max<size_t>(1, n_req));
    for (size_t i = 0; i <= n_req; ++i) {
        input_offsets_vec[i] = static_cast<int32_t>(std::min<size_t>(bucket, i * per_req));
    }
    input_offsets_vec[n_req] = static_cast<int32_t>(bucket);
    infinicore::context::memcpyH2D(
        input.input_offsets.value()->data(),
        input_offsets_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);

    input.cu_seqlens = infinicore::Tensor::empty({n_req + 1}, infinicore::DataType::I32, device);
    std::vector<int32_t> cu_seqlens_vec(n_req + 1, 0);
    for (size_t i = 0; i <= n_req; ++i) {
        cu_seqlens_vec[i] = static_cast<int32_t>(std::min<size_t>(bucket, i * per_req));
    }
    cu_seqlens_vec[n_req] = static_cast<int32_t>(bucket);
    infinicore::context::memcpyH2D(
        input.cu_seqlens.value()->data(),
        cu_seqlens_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);

    const size_t block_per_req = nblocks;
    input.block_tables = block_tables_holder_->as_strided({n_req, block_per_req}, {(ptrdiff_t)block_per_req, 1});
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const size_t block_size = paged_config != nullptr ? paged_config->block_size() : 256;
    const size_t blocks_needed = (bucket + block_size - 1) / block_size;
    for (size_t row = 0; row < n_req; ++row) {
        std::vector<int32_t> block_row(block_per_req, -1);
        const size_t row_offset = row * blocks_needed;
        for (size_t b = 0; b < blocks_needed && b < block_per_req; ++b) {
            block_row[b] = static_cast<int32_t>(row_offset + b);
        }
        auto row_tensor = input.block_tables.value()->narrow({{0, row, 1}});
        infinicore::context::memcpyH2D(
            row_tensor->data(), block_row.data(), block_per_req * sizeof(int32_t), false);
    }

    input.slot_mapping = infinicore::Tensor::empty({bucket}, infinicore::DataType::I64, device);
    std::vector<int64_t> slot_mapping_vec(bucket);
    std::iota(slot_mapping_vec.begin(), slot_mapping_vec.end(), int64_t{0});
    infinicore::context::memcpyH2D(
        input.slot_mapping.value()->data(), slot_mapping_vec.data(), bucket * sizeof(int64_t), false);
    // #region agent log
    infinilm::agent_debug::log(
        "piecewise_prefill_compiler.cpp:make_bucket_input_",
        "capture_input_shape",
        "C",
        "{\"bucket\":" + std::to_string(bucket) + ",\"n_req\":" + std::to_string(n_req) +
            ",\"per_req_tokens\":" + std::to_string(bucket / std::max<size_t>(1, n_req)) +
            ",\"slot_mapping_len\":" + std::to_string(bucket) + ",\"block_per_req\":" +
            std::to_string(block_per_req) + "}",
        "repro");
    // #endregion
    return input;
}

void PiecewisePrefillCompiler::capture_bucket_(size_t bucket) {
    auto &piecewise_flag = infinilm::global_state::get_forward_context().piecewise;
    struct CaptureGuard {
        infinilm::global_state::PiecewisePrefillState &pw;
        explicit CaptureGuard(infinilm::global_state::PiecewisePrefillState &p) : pw(p) { pw.compile_capture_active = true; }
        ~CaptureGuard() { pw.compile_capture_active = false; }
    } capture_guard(piecewise_flag);

    const size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
    const size_t num_layers = model_->native_piecewise_num_layers();
    // #region agent log
    {
        const auto dev = infinicore::context::getDevice().toString();
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "piecewise_prefill_compiler.cpp:capture_bucket_",
            "capture_begin",
            "A",
            "{\"bucket\":" + std::to_string(bucket) + ",\"max_capture_req\":" +
                std::to_string(max_capture_req_) + ",\"nblocks\":" + std::to_string(nblocks) +
                ",\"device\":\"" + dev + "\"}",
            "repro");
        infinilm::agent_debug::session_log(
            "piecewise_prefill_compiler.cpp:capture_bucket_",
            "capture_begin",
            "H3",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"bucket\":" +
                std::to_string(bucket) + ",\"max_capture_req\":" +
                std::to_string(max_capture_req_) + ",\"max_seq\":" +
                std::to_string(max_seq_len_) + "}");
    }
    // #endregion
    allocate_layer_staging_(bucket, num_layers);
    auto bucket_input = make_bucket_input_(bucket, nblocks, max_capture_req_);
    set_attn_metadata(bucket_input);

    BucketGraphs graphs;
    graphs.input = std::move(bucket_input);

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = bucket;
    piecewise.phase = global_state::PiecewiseCapturePhase::None;

    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;

    size_t capture_layers = num_layers;
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_MAX_LAYERS")) {
        capture_layers = std::min(num_layers, static_cast<size_t>(std::stoul(raw)));
    }

    const auto &model_config = infinilm::global_state::get_infinilm_config().model_config;
    const auto dtype = model_config->get_dtype();
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    graphs.logits_holder = infinicore::Tensor::empty(
        {1, bucket, vocab_size}, dtype, infinicore::context::getDevice());

    // Eager warmup dry-run before capture (NONE-equivalent).
    model_->native_piecewise_embed(graphs.input, hidden);
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
    }
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    graphs.pre_attn.resize(capture_layers);
    graphs.post_attn.resize(capture_layers);

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(graphs.input, hidden);

    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;

        barrier_->wait("piecewise_capture_pre_attn");
        // #region agent log
        if (layer == 0) {
            const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
            infinilm::agent_debug::session_log(
                "piecewise_prefill_compiler.cpp:capture_bucket_",
                "graph_record_pre_attn",
                "H5",
                std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"bucket\":" +
                    std::to_string(bucket) + ",\"layer\":0}");
        }
        // #endregion
        infinicore::context::startGraphRecording();
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        graphs.pre_attn[layer] = infinicore::context::stopGraphRecording();
        infinicore::context::syncStream();

        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        barrier_->wait("piecewise_capture_post_attn");
        infinicore::context::startGraphRecording();
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        graphs.post_attn[layer] = infinicore::context::stopGraphRecording();
        barrier_->wait("piecewise_capture_post_attn_sync");
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
    barrier_->wait("piecewise_capture_lm_head");
    infinicore::context::startGraphRecording();
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    graphs.lm_head = infinicore::context::stopGraphRecording();
    barrier_->wait("piecewise_capture_lm_head_sync");

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    graphs.hidden_states = piecewise.hidden_states;
    graphs.residual = piecewise.residual;
    graphs.ar_staging = piecewise.ar_staging;
    graphs.layer_staging = piecewise.layer_staging;
    compiled_[bucket] = std::move(graphs);
    spdlog::info("native piecewise CG: captured bucket={} layers={} segments={}",
                 bucket, capture_layers, capture_layers * 2 + 1);
    // #region agent log
    infinilm::agent_debug::log(
        "piecewise_prefill_compiler.cpp:capture_bucket_",
        "capture_done",
        "A",
        "{\"bucket\":" + std::to_string(bucket) + ",\"layers\":" + std::to_string(capture_layers) + "}",
        "repro");
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::session_log(
            "piecewise_prefill_compiler.cpp:capture_bucket_",
            "capture_done",
            "H5",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"bucket\":" +
                std::to_string(bucket) + ",\"layers\":" + std::to_string(capture_layers) + "}");
    }
    // #endregion
}

void PiecewisePrefillCompiler::compile() {
    if (!native_piecewise_prefill_enabled() || !model_->supports_native_piecewise_prefill()) {
        enabled_ = false;
        return;
    }
    if (model_->get_cache_config() == nullptr
        || dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config()) == nullptr) {
        spdlog::debug("PiecewisePrefillCompiler: defer capture until paged cache is configured");
        return;
    }
    enabled_ = true;

    const auto *paged_config =
        dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const size_t nblocks = paged_config->num_blocks();
    max_capture_req_ = std::max<size_t>(1, paged_config->max_batch_size());
    if (const char *raw = std::getenv("INFINI_MAX_PREFILL_BATCH")) {
        max_capture_req_ = std::max<size_t>(1, std::stoul(raw));
    }
    size_t max_bucket = capture_buckets_.empty() ? 0 : capture_buckets_.front();
    block_tables_holder_ = infinicore::Tensor::empty(
        {max_capture_req_ * nblocks}, infinicore::DataType::I32, infinicore::context::getDevice());
    set_zeros(block_tables_holder_);
    spdlog::info(
        "native piecewise CG: capture warmup n_req={} (metadata only, hidden [1,bucket])",
        max_capture_req_);

    compiled_.clear();
    for (size_t bucket : capture_buckets_) {
        capture_bucket_(bucket);
        infinicore::context::syncDevice();
    }
    std::ostringstream oss;
    for (size_t i = 0; i < capture_buckets_.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << capture_buckets_[i];
    }
    spdlog::info("native piecewise CG: capture_buckets=[{}] max_seq={}",
                 oss.str(), max_seq_len_);
    // #region agent log
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "piecewise_prefill_compiler.cpp:compile",
            "compile_complete",
            "P",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"n_buckets\":" +
                std::to_string(capture_buckets_.size()) + ",\"max_bucket\":" +
                std::to_string(max_bucket) + "}",
            "post-fix");
    }
    // #endregion
}

size_t PiecewisePrefillCompiler::padded_bucket_for(size_t seq_len) const {
    return padded_bucket_for_seq_len(seq_len, bs_to_padded_, max_seq_len_);
}

void PiecewisePrefillCompiler::copy_runtime_into_bucket_(BucketGraphs &bucket_graphs,
                                                         const InfinilmModel::Input &runtime,
                                                         size_t valid_seq_len) const {
    const size_t bucket = bucket_graphs.input.input_ids.value()->size(1);
    auto &graph_input = bucket_graphs.input;

    graph_input.input_ids.value()
        ->narrow({{1, 0, valid_seq_len}})
        ->copy_from(runtime.input_ids.value());
    graph_input.position_ids.value()
        ->narrow({{0, 0, valid_seq_len}})
        ->copy_from(runtime.position_ids.value());

    const size_t runtime_n_req = runtime.block_tables.value()->size(0);
    const size_t compiled_n_req = graph_input.block_tables.value()->size(0);
    if (runtime_n_req > compiled_n_req) {
        throw std::runtime_error("block_tables batch exceeds compiled capture warmup width");
    }

    if (graph_input.past_sequence_lengths.has_value() && runtime.past_sequence_lengths.has_value()) {
        graph_input.past_sequence_lengths.value()
            ->narrow({{0, 0, runtime_n_req}})
            ->copy_from(runtime.past_sequence_lengths.value());
    }
    graph_input.total_sequence_lengths.value()
        ->narrow({{0, 0, runtime_n_req}})
        ->copy_from(runtime.total_sequence_lengths.value());
    graph_input.input_offsets.value()
        ->narrow({{0, 0, runtime_n_req + 1}})
        ->copy_from(runtime.input_offsets.value());
    graph_input.cu_seqlens.value()
        ->narrow({{0, 0, runtime_n_req + 1}})
        ->copy_from(runtime.cu_seqlens.value());

    const size_t block_per_req = runtime.block_tables.value()->size(1);
    const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
    if (block_per_req > compiled_block_per_req) {
        throw std::runtime_error("block_tables width exceeds compiled bucket");
    }

    auto &graph_block_tables = graph_input.block_tables.value();
    set_minus_one(graph_block_tables);
    graph_block_tables
        ->narrow({{0, 0, runtime_n_req}, {1, 0, block_per_req}})
        ->copy_from(runtime.block_tables.value());

    if (runtime_n_req < compiled_n_req) {
        auto stale_rows = graph_block_tables->narrow(
            {{0, runtime_n_req, compiled_n_req - runtime_n_req}, {1, 0, compiled_block_per_req}});
        set_minus_one(stale_rows);
    }

    graph_input.slot_mapping.value()
        ->narrow({{0, 0, valid_seq_len}})
        ->copy_from(runtime.slot_mapping.value());
    if (valid_seq_len < bucket) {
        auto slot_tail = graph_input.slot_mapping.value()->narrow({{0, valid_seq_len, bucket - valid_seq_len}});
        set_minus_one(slot_tail);
        auto ids_tail = graph_input.input_ids.value()->narrow({{1, valid_seq_len, bucket - valid_seq_len}});
        set_zeros(ids_tail);
        auto pos_tail = graph_input.position_ids.value()->narrow({{0, valid_seq_len, bucket - valid_seq_len}});
        set_zeros(pos_tail);
    }
}

std::optional<infinicore::Tensor> PiecewisePrefillCompiler::run_prefill(const InfinilmModel::Input &input) {
    if (!enabled_) {
        return std::nullopt;
    }
    const bool profile = rank_worker_profile_enabled();
    const double t_total0 = profile ? monotonic_ms() : 0.0;
    const size_t seq_len = compute_prefill_len(input);
    const size_t padded = padded_bucket_for(seq_len);
    const size_t graph_bucket = graph_replay_bucket_for_padded(padded);
    const size_t runtime_n_req = input.block_tables.has_value() ? input.block_tables.value()->size(0) : 1;
    const bool final_chunk = InfinilmModel::any_final_prefill_chunk(input.is_final_prefill_chunk);
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise run_prefill begin seq_len={} padded={} graph_bucket={} "
            "n_req={} final_chunk={}",
            seq_len,
            padded,
            graph_bucket,
            runtime_n_req,
            final_chunk);
    }
    if (compiled_.find(graph_bucket) == compiled_.end()) {
        ++prefill_misses_;
        return std::nullopt;
    }
    // Bucket graphs are captured at full graph_bucket width; replay with shorter
    // valid_seq_len leaves stale tail state in captured segments. Fall back to
    // eager forward for partial buckets (matches tp=4 eager path for short prompts).
    if (seq_len < graph_bucket) {
        ++prefill_misses_;
        // #region agent log
        if (infinilm::global_state::get_tensor_model_parallel_rank() == 0) {
            infinilm::agent_debug::log(
                "piecewise_prefill_compiler.cpp:run_prefill",
                "pw_fallback_eager",
                "Z1",
                std::string("{\"seq_len\":") + std::to_string(seq_len) + ",\"graph_bucket\":" +
                    std::to_string(graph_bucket) + "}",
                "post-fix");
        }
        // #endregion
        return std::nullopt;
    }

    // #region agent log
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "piecewise_prefill_compiler.cpp:run_prefill",
            "pw_replay_begin",
            "P3",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"n_req\":"
                + std::to_string(runtime_n_req) + ",\"seq_len\":" + std::to_string(seq_len)
                + ",\"graph_bucket\":" + std::to_string(graph_bucket) + "}",
            "repro");
    }
    // #endregion
    auto &bucket_graphs = compiled_.at(graph_bucket);
    const double t_copy0 = profile ? monotonic_ms() : 0.0;
    copy_runtime_into_bucket_(bucket_graphs, input, seq_len);
    set_attn_metadata_for_varlen_batch(bucket_graphs.input, input);
    // #region agent log
    {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "piecewise_prefill_compiler.cpp:run_prefill",
            "pw_replay_dims",
            "W2",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"seq_len\":" +
                std::to_string(seq_len) + ",\"padded\":" + std::to_string(padded) +
                ",\"graph_bucket\":" + std::to_string(graph_bucket) + "}",
            graph_bucket == 256 ? "bucket-256-smoke" : "meta-check");
    }
    // #endregion
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise copy_runtime+metadata_ms={:.3f}",
            monotonic_ms() - t_copy0);
    }

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = seq_len;
    piecewise.bucket_seq_len = padded;
    piecewise.hidden_states = bucket_graphs.hidden_states;
    piecewise.residual = bucket_graphs.residual;
    piecewise.ar_staging = bucket_graphs.ar_staging;
    piecewise.layer_staging = bucket_graphs.layer_staging;

    // Fresh residual each replay (matches capture warmup); hidden prefix comes from embed.
    set_zeros(piecewise.residual);
    if (seq_len < graph_bucket) {
        clear_stale_bucket_tails_(piecewise, bucket_graphs.logits_holder, seq_len, graph_bucket);
    }

    model_->native_piecewise_embed(bucket_graphs.input, piecewise.hidden_states);

    const size_t num_layers = bucket_graphs.pre_attn.size();
    const bool use_eager_post = seq_len < graph_bucket;
    const bool use_eager_lm_head = seq_len < graph_bucket;
    // #region agent log
    if (infinilm::agent_debug::debug_enabled()) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "piecewise_prefill_compiler.cpp:run_prefill",
            "pw_replay_mode",
            "Z5",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"seq_len\":" +
                std::to_string(seq_len) + ",\"graph_bucket\":" + std::to_string(graph_bucket) +
                ",\"use_eager_post\":" + (use_eager_post ? "true" : "false") +
                ",\"use_eager_lm_head\":" + (use_eager_lm_head ? "true" : "false") +
                ",\"segment_replays\":" + std::to_string(segment_replays_) + "}",
            "rc7");
    }
    // #endregion
    double layers_ms = 0.0;
    const double t_layers0 = profile ? monotonic_ms() : 0.0;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        const double t_layer0 = profile ? monotonic_ms() : 0.0;
        barrier_->wait("piecewise_replay_pre_attn");
        if (use_eager_post) {
            model_->native_piecewise_pre_attn_layer(
                layer, bucket_graphs.input, piecewise.hidden_states, piecewise.residual);
        } else {
            bucket_graphs.pre_attn[layer]->run();
            ++segment_replays_;
        }
        const double t_pre_attn = profile ? monotonic_ms() : 0.0;
        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, bucket_graphs.input);
        const double t_eager_attn = profile ? monotonic_ms() : 0.0;
        barrier_->wait("piecewise_replay_post_attn");
        if (use_eager_post) {
            model_->native_piecewise_post_attn_cg_layer(
                layer, bucket_graphs.input, piecewise.hidden_states, piecewise.residual);
            barrier_->wait("piecewise_replay_post_attn_sync");
        } else {
            bucket_graphs.post_attn[layer]->run();
            ++segment_replays_;
            barrier_->wait("piecewise_replay_post_attn_sync");
        }
        // #region agent log
        if (layer <= 1) {
            const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
            infinilm::agent_debug::log(
                "piecewise_prefill_compiler.cpp:run_prefill",
                "pw_hidden_after_layer",
                "Y2",
                std::string("{\"tp_rank\":") + std::to_string(tp_rank) + ",\"layer\":" +
                    std::to_string(layer) + ",\"use_eager_post\":" + (use_eager_post ? "true" : "false") +
                    ",\"first_bits\":" +
                    std::to_string(infinilm::agent_debug::first_elem_bits(
                        piecewise.hidden_states->narrow({{1, 0, seq_len}}))) +
                    "}",
                "rc7");
        }
        // #endregion
        if (profile) {
            spdlog::info(
                "rank_worker_profile: piecewise layer={} pre_attn_ms={:.3f} eager_attn_ms={:.3f} "
                "post_attn_ms={:.3f} layer_total_ms={:.3f}",
                layer,
                t_pre_attn - t_layer0,
                t_eager_attn - t_pre_attn,
                monotonic_ms() - t_eager_attn,
                monotonic_ms() - t_layer0);
        }
    }
    if (profile) {
        layers_ms = monotonic_ms() - t_layers0;
    }
    if (InfinilmModel::any_final_prefill_chunk(input.is_final_prefill_chunk)) {
        const double t_lm0 = profile ? monotonic_ms() : 0.0;
        barrier_->wait("piecewise_replay_lm_head");
        if (use_eager_lm_head) {
            model_->native_piecewise_lm_head(
                bucket_graphs.input, piecewise.hidden_states, piecewise.residual, bucket_graphs.logits_holder);
        } else {
            bucket_graphs.lm_head->run();
            ++segment_replays_;
        }
        barrier_->wait("piecewise_replay_lm_head_sync");
        if (profile) {
            spdlog::info(
                "rank_worker_profile: piecewise lm_head_ms={:.3f}",
                monotonic_ms() - t_lm0);
        }
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    ++prefill_hits_;
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise run_prefill end layers_total_ms={:.3f} total_ms={:.3f}",
            layers_ms,
            monotonic_ms() - t_total0);
    }
    if (global_state::ar_profile::enabled()
        && global_state::get_tensor_model_parallel_rank() == 0) {
        global_state::ar_profile::log_barrier_chunk_summary(
            "piecewise_replay", seq_len, runtime_n_req);
    }
    if (!InfinilmModel::any_final_prefill_chunk(input.is_final_prefill_chunk)) {
        return std::nullopt;
    }
    return bucket_graphs.logits_holder;
}

} // namespace infinilm::engine
