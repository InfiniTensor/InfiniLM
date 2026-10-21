#include "piecewise_prefill_compiler.hpp"

#include "../../global_state/ar_profile.hpp"
#include "../../global_state/global_state.hpp"
#include "../../global_state/hang_trace.hpp"
#include "../compiled_prefill_flags.hpp"
#include "../../global_state/piecewise_inductor_flags.hpp"
#include "../../utils.hpp"
#include "../dispatch_hist.hpp"
#include "piecewise_bucket_policy.hpp"
#include "attn_metadata_utils.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <spdlog/spdlog.h>
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"
#include "infinicore/ops/inductor_segment.hpp"

namespace infinilm::engine {

namespace {

bool graph_capture_audit_enabled_() {
    const char *v = std::getenv("INFINI_GRAPH_CAPTURE_AUDIT");
    return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

/// When MoE is device-capturable, post_attn must be a single hcGraph segment
/// (true one-piece: o_proj+AR+LN+MoE). Fail fast under capture audit.
/// FORCE_OP_LIST skips device capture on purpose → device_segments=0 is OK.
void assert_post_one_device_seg_(
    const char *where,
    size_t layer,
    const std::shared_ptr<infinicore::graph::Graph> &g) {
    if (!graph_capture_audit_enabled_()
        || !infinicore::context::moeTritonCaptureAllowed()
        || !g) {
        return;
    }
    if (const char *fol = std::getenv("INFINI_GRAPH_FORCE_OP_LIST");
        fol != nullptr && fol[0] != '\0' && std::string(fol) != "0") {
        return;
    }
    const size_t n = g->device_segment_count();
    if (n == 1) {
        return;
    }
    spdlog::error(
        "[capture_audit] {} layer={} post device_segments={} expected=1 "
        "(MoE capturable — true one-piece post requires a single device seg)",
        where,
        layer,
        n);
    throw std::runtime_error(
        std::string("[capture_audit] post device_segments!=1 when MoE capturable (")
        + where + ")");
}

bool rank_worker_profile_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_RANK_WORKER_PROFILE");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// When set with RANK_WORKER_PROFILE, syncDevice after each Graph::run so
/// *_graph_run_ms is exclusive GPU+host (mirrors decode_phase_profile exclusive_sync).
bool rank_worker_profile_sync() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_RANK_WORKER_PROFILE_SYNC");
        cached = (raw != nullptr && raw[0] == '1' && raw[1] == '\0') ? 1 : 0;
    }
    return cached == 1;
}

/// Prefill TP=1: Graph::run may skip its end syncStream (INFINI_GRAPH_SKIP_TAIL_SYNC).
/// Callers must drain once before logits / sample D2H (Gate C).
bool graph_skip_tail_sync_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *raw = std::getenv("INFINI_GRAPH_SKIP_TAIL_SYNC");
        cached = (raw != nullptr && raw[0] != '\0' && std::string(raw) != "0") ? 1 : 0;
    }
    return cached == 1;
}

void append_graph_run_split(
    std::string &out,
    const char *prefix,
    const infinicore::graph::Graph::LastRunProfile &p) {
    char buf[256];
    std::snprintf(
        buf,
        sizeof(buf),
        " %s_launch_ms=%.3f %s_tail_sync_ms=%.3f %s_hostop_ms=%.3f %s_op_list_ms=%.3f",
        prefix,
        p.launch_ms,
        prefix,
        p.tail_sync_ms,
        prefix,
        p.hostop_ms,
        prefix,
        p.op_list_ms);
    out.append(buf);
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

bool repro_skip_midchunk_eager() {
    return infinilm::global_state::repro_skip_midchunk_eager();
}

bool repro_allow_midchunk_cg() {
    return infinilm::global_state::repro_allow_midchunk_cg();
}

bool repro_mixed_mid_cg() {
    return infinilm::global_state::repro_mixed_mid_cg();
}

bool enable_mixed_mid_cg() {
    return infinilm::global_state::enable_mixed_mid_cg();
}

bool keep_mixed_mid_capture() {
    return infinilm::global_state::keep_mixed_mid_capture();
}

/// Skip homo mid-chunk dual capture (saves ~2x device-graph execs). Default off;
/// QY KS38 sets INFINI_NATIVE_CG_SKIP_MID_CAPTURE=1 from run-wrap.
bool skip_homo_mid_capture() {
    const char *raw = std::getenv("INFINI_NATIVE_CG_SKIP_MID_CAPTURE");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

bool scoped_inductor_pre_attn() {
    return infinilm::global_state::scoped_inductor_pre_attn_enabled();
}

bool repro_skip_final_inductor() {
    return infinilm::global_state::repro_skip_final_inductor();
}

bool pre_attn_qkv_only() {
    return infinilm::global_state::pre_attn_qkv_only();
}

bool piecewise_pad_up_cg_enabled() {
    return infinilm::global_state::piecewise_pad_up_cg_enabled();
}

bool bucket_inductor_capture_enabled(size_t bucket) {
    return infinilm::global_state::piecewise_inductor_segment_enabled()
           && infinilm::global_state::scoped_inductor_pre_attn_enabled()
           && infinilm::global_state::bucket_is_inductor_eligible(bucket);
}

bool layer_capture_inductor_pre_attn(size_t layer, size_t bucket) {
    return bucket_inductor_capture_enabled(bucket)
           && infinicore::op::inductor_segment_impl::has_package(
               infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer, bucket);
}

void verify_inductor_packages_(const std::shared_ptr<InfinilmModel> &model,
                               const std::vector<size_t> &capture_buckets) {
    if (!infinilm::global_state::piecewise_inductor_segment_enabled()) {
        return;
    }
    if (!infinilm::global_state::scoped_inductor_pre_attn_enabled()) {
        return;
    }
    const size_t num_layers = model->native_piecewise_num_layers();
    const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    for (size_t bucket : capture_buckets) {
        if (!infinilm::global_state::bucket_is_inductor_eligible(bucket)) {
            continue;
        }
        for (size_t layer = 0; layer < num_layers; ++layer) {
            if (!infinicore::op::inductor_segment_impl::has_package(
                    infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer, bucket)) {
                throw std::runtime_error(
                    "piecewise inductor: missing AOT pre_attn package layer="
                    + std::to_string(layer) + " bucket=" + std::to_string(bucket)
                    + " tp_rank=" + std::to_string(tp_rank)
                    + " (compile with aot_compile_piecewise_segments.py --buckets "
                    + std::to_string(bucket) + " --layers all)");
            }
        }
    }
}

double monotonic_ms() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clock::now().time_since_epoch()).count();
}

size_t device_free_bytes_() {
    // Post-refactor InfiniRT does not expose the legacy infinirtGetMemInfo
    // path used for MetaX capture audit; return 0 (logging only).
    (void)infinicore::context::getDevice();
    return 0;
}

size_t tensor_nbytes_(const infinicore::Tensor &t) {
    return t ? t->nbytes() : 0;
}

size_t graph_arena_nbytes_(const std::shared_ptr<infinicore::graph::Graph> &g) {
    return g ? g->capture_arena_bytes() : 0;
}

size_t bucket_arena_nbytes_(const PiecewisePrefillCompiler::BucketGraphs &g) {
    size_t n = graph_arena_nbytes_(g.lm_head);
    for (const auto &seg : g.pre_attn) {
        n += graph_arena_nbytes_(seg);
    }
    for (const auto &seg : g.post_attn) {
        n += graph_arena_nbytes_(seg);
    }
    return n;
}

double bytes_to_gib_(size_t n) {
    return static_cast<double>(n) / (1024.0 * 1024.0 * 1024.0);
}

size_t compute_prefill_len(const InfinilmModel::Input &input) {
    if (input.input_offsets.has_value()) {
        const auto &offsets = input.input_offsets.value();
        const size_t n = offsets->size(0);
        if (n >= 2) {
            auto cpu_offsets = offsets->to(infinicore::Device{infinicore::Device::Type::kCpu});
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
    attn_metadata_utils::set_attn_metadata(input);
}

void set_attn_metadata_for_varlen_batch(const InfinilmModel::Input &compiled,
                                        const InfinilmModel::Input &runtime) {
    attn_metadata_utils::set_attn_metadata_for_varlen_batch(compiled, runtime);
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
    prefill_chunk_size_ = prefill_chunk_size_from_env();
    const size_t chunk_cap = prefill_chunk_size_;
    const bool vllm_ladder = vllm_capture_ladder_enabled();
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
        // Runtime pad ladder: keep full vLLM ladder when enabled; capture list may be a subset
        // (e.g. Qwen3-32B captures B8192 only to avoid multi-bucket CG OOM under TP=4).
        if (vllm_ladder) {
            bs_to_padded_ = build_bs_to_padded_bucket(
                piecewise_compile_buckets_vllm(max_seq_len_, chunk_cap));
        } else {
            bs_to_padded_ = build_bs_to_padded_bucket(capture_buckets_);
        }
    } else if (vllm_ladder) {
        capture_buckets_ = piecewise_capture_buckets_vllm(max_seq_len_, chunk_cap);
        bs_to_padded_ = build_bs_to_padded_bucket(capture_buckets_);
    } else {
        auto pad_ladder = piecewise_compile_buckets(max_seq_len_);
        bs_to_padded_ = build_bs_to_padded_bucket(pad_ladder);
        capture_buckets_ = piecewise_capture_buckets(max_seq_len_);
    }
    std::sort(capture_buckets_.begin(), capture_buckets_.end(), std::greater<size_t>());
}

void PiecewisePrefillCompiler::allocate_shared_banks_(size_t max_bucket,
                                                      size_t num_layers,
                                                      size_t n_req) {
    shared_banks_ = SharedPrefillBanks{};
    shared_banks_.max_bucket = max_bucket;
    shared_banks_.num_layers = num_layers;
    const auto device = infinicore::context::getDevice();
    const auto &model_config = infinilm::global_state::get_infinilm_config().model_config;
    const auto dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");
    const size_t tp_size = std::max<size_t>(
        1, static_cast<size_t>(infinilm::global_state::get_tensor_model_parallel_world_size()));
    const size_t num_heads = model_config->get<size_t>("num_attention_heads") / static_cast<size_t>(tp_size);
    const size_t total_kv = model_config->get<size_t>("num_key_value_heads");
    const size_t num_kv_heads = total_kv < tp_size ? 1 : total_kv / tp_size;
    const size_t head_dim = model_config->get_head_dim();

    shared_banks_.layer_staging.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        auto &st = shared_banks_.layer_staging[i];
        st.q_rope = infinicore::Tensor::empty({1, max_bucket, num_heads, head_dim}, dtype, device);
        st.k_rope = infinicore::Tensor::empty({1, max_bucket, num_kv_heads, head_dim}, dtype, device);
        st.v_rope = infinicore::Tensor::empty({1, max_bucket, num_kv_heads, head_dim}, dtype, device);
        st.attn_output = infinicore::Tensor::empty({1, max_bucket, num_heads * head_dim}, dtype, device);
    }
    shared_banks_.hidden_states = infinicore::Tensor::empty({1, max_bucket, hidden}, dtype, device);
    shared_banks_.residual = infinicore::Tensor::empty({1, max_bucket, hidden}, dtype, device);
    shared_banks_.ar_staging = infinicore::Tensor::empty({1, max_bucket, hidden}, dtype, device);

    shared_banks_.input_ids =
        infinicore::Tensor::empty({1, max_bucket}, infinicore::DataType::kInt64, device);
    shared_banks_.position_ids =
        infinicore::Tensor::empty({max_bucket}, infinicore::DataType::kInt64, device);
    shared_banks_.slot_mapping =
        infinicore::Tensor::empty({max_bucket}, infinicore::DataType::kInt64, device);
    shared_banks_.past_sequence_lengths =
        infinicore::Tensor::empty({n_req}, infinicore::DataType::kInt32, device);
    shared_banks_.total_sequence_lengths =
        infinicore::Tensor::empty({n_req}, infinicore::DataType::kInt32, device);
    shared_banks_.input_offsets =
        infinicore::Tensor::empty({n_req + 1}, infinicore::DataType::kInt32, device);
    shared_banks_.cu_seqlens =
        infinicore::Tensor::empty({n_req + 1}, infinicore::DataType::kInt32, device);
    shared_banks_.logits_holder =
        infinicore::Tensor::empty({1, max_bucket, vocab_size}, dtype, device);

    size_t staging = tensor_nbytes_(shared_banks_.hidden_states)
                     + tensor_nbytes_(shared_banks_.residual)
                     + tensor_nbytes_(shared_banks_.ar_staging);
    for (const auto &st : shared_banks_.layer_staging) {
        staging += tensor_nbytes_(st.q_rope) + tensor_nbytes_(st.k_rope)
                   + tensor_nbytes_(st.v_rope) + tensor_nbytes_(st.attn_output);
    }
    const size_t io = tensor_nbytes_(shared_banks_.input_ids)
                      + tensor_nbytes_(shared_banks_.position_ids)
                      + tensor_nbytes_(shared_banks_.slot_mapping)
                      + tensor_nbytes_(shared_banks_.past_sequence_lengths)
                      + tensor_nbytes_(shared_banks_.total_sequence_lengths)
                      + tensor_nbytes_(shared_banks_.input_offsets)
                      + tensor_nbytes_(shared_banks_.cu_seqlens)
                      + tensor_nbytes_(shared_banks_.logits_holder);
    shared_banks_.staging_physical_bytes = staging;
    shared_banks_.io_physical_bytes = io;
    spdlog::info(
        "native piecewise CG: shared max-bucket banks max_bucket={} layers={} "
        "staging_GiB={:.3f} io_GiB={:.3f}",
        max_bucket,
        num_layers,
        bytes_to_gib_(staging),
        bytes_to_gib_(io));
}

void PiecewisePrefillCompiler::bind_bucket_staging_(size_t bucket, size_t num_layers) {
    if (!shared_banks_.hidden_states || bucket > shared_banks_.max_bucket) {
        throw std::runtime_error(
            "bind_bucket_staging_: shared banks missing or bucket exceeds max");
    }
    if (num_layers > shared_banks_.num_layers) {
        throw std::runtime_error("bind_bucket_staging_: num_layers exceeds shared bank");
    }
    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.bucket_seq_len = bucket;
    piecewise.layer_staging.clear();
    piecewise.layer_staging.resize(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        const auto &src = shared_banks_.layer_staging[i];
        auto &st = piecewise.layer_staging[i];
        st.q_rope = src.q_rope->narrow({{1, 0, bucket}});
        st.k_rope = src.k_rope->narrow({{1, 0, bucket}});
        st.v_rope = src.v_rope->narrow({{1, 0, bucket}});
        st.attn_output = src.attn_output->narrow({{1, 0, bucket}});
    }
    piecewise.hidden_states = shared_banks_.hidden_states->narrow({{1, 0, bucket}});
    piecewise.residual = shared_banks_.residual->narrow({{1, 0, bucket}});
    piecewise.ar_staging = shared_banks_.ar_staging->narrow({{1, 0, bucket}});
}

InfinilmModel::Input PiecewisePrefillCompiler::make_bucket_input_(size_t bucket,
                                                                  size_t nblocks,
                                                                  size_t n_req,
                                                                  bool mid_chunk_capture) const {
    if (!shared_banks_.input_ids || bucket > shared_banks_.max_bucket) {
        throw std::runtime_error("make_bucket_input_: shared I/O banks not allocated");
    }
    InfinilmModel::Input input;
    input.input_ids = shared_banks_.input_ids->narrow({{1, 0, bucket}});
    input.position_ids = shared_banks_.position_ids->narrow({{0, 0, bucket}});
    input.past_sequence_lengths = shared_banks_.past_sequence_lengths;
    input.total_sequence_lengths = shared_banks_.total_sequence_lengths;
    input.input_offsets = shared_banks_.input_offsets;
    input.cu_seqlens = shared_banks_.cu_seqlens;
    input.slot_mapping = shared_banks_.slot_mapping->narrow({{0, 0, bucket}});
    set_zeros(input.input_ids.value());
    set_zeros(input.past_sequence_lengths.value());
    set_zeros(input.total_sequence_lengths.value());

    const size_t chunk_size =
        prefill_chunk_size_ > 0 ? prefill_chunk_size_ : prefill_chunk_size_from_env();
    // Mid dual-capture: representative continuing chunk after one full prior chunk.
    const size_t past =
        mid_chunk_capture ? (bucket == chunk_size ? bucket : chunk_size) : 0;
    const int64_t pos_start = mid_chunk_capture
        ? static_cast<int64_t>(past)
        : ((bucket < chunk_size) ? static_cast<int64_t>(chunk_size) : int64_t{0});
    std::vector<int64_t> position_ids_vec(bucket);
    std::iota(position_ids_vec.begin(), position_ids_vec.end(), pos_start);
    infinicore::context::memcpyH2D(
        input.position_ids.value()->data(), position_ids_vec.data(), bucket * sizeof(int64_t), false);

    const int32_t per_req = static_cast<int32_t>(bucket / std::max<size_t>(1, n_req));
    std::vector<int32_t> past_lengths_vec(n_req, static_cast<int32_t>(past));
    std::vector<int32_t> total_lengths_vec(
        n_req, static_cast<int32_t>(past) + per_req);
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

    std::vector<int32_t> input_offsets_vec(n_req + 1, 0);
    for (size_t i = 0; i <= n_req; ++i) {
        input_offsets_vec[i] = static_cast<int32_t>(std::min<size_t>(bucket, i * per_req));
    }
    input_offsets_vec[n_req] = static_cast<int32_t>(bucket);
    infinicore::context::memcpyH2D(
        input.input_offsets.value()->data(),
        input_offsets_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);

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
    const size_t span_tokens = past + bucket;
    const size_t blocks_needed = (span_tokens + block_size - 1) / block_size;
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

    std::vector<int64_t> slot_mapping_vec(bucket);
    std::iota(slot_mapping_vec.begin(),
              slot_mapping_vec.end(),
              static_cast<int64_t>(past));
    infinicore::context::memcpyH2D(
        input.slot_mapping.value()->data(), slot_mapping_vec.data(), bucket * sizeof(int64_t), false);
    // Mid capture is a continuing chunk (not final); leave flag empty → treated as final for
    // lm_head dry-run only when we explicitly run lm_head below. Runtime mid uses is_final=false.
    return input;
}

InfinilmModel::Input PiecewisePrefillCompiler::make_mixed_mid_bucket_input_(size_t bucket,
                                                                           size_t nblocks) const {
    if (!shared_banks_.input_ids || bucket > shared_banks_.max_bucket) {
        throw std::runtime_error("make_mixed_mid_bucket_input_: shared I/O banks not allocated");
    }
    if (bucket < 2) {
        throw std::runtime_error("make_mixed_mid_bucket_input_: bucket must be >= 2");
    }
    const size_t n_req = mixed_mid_capture_req_;
    // Match v1 scheduler RUNNING order: decode rows often precede mid-prefills.
    // Dominant LongBench MIXED@2048: 1 decode token + continuing mid fills bucket-1.
    const size_t decode_q = 1;
    const size_t mid_q = bucket - 1;
    const size_t decode_past = bucket;
    const size_t mid_past = bucket;
    const size_t decode_total = decode_past + decode_q;
    const size_t mid_total = mid_past + mid_q;

    InfinilmModel::Input input;
    input.input_ids = shared_banks_.input_ids->narrow({{1, 0, bucket}});
    input.position_ids = shared_banks_.position_ids->narrow({{0, 0, bucket}});
    // Narrow metadata to capture n_req so FA max_seqlens do not see zero-pad tails.
    input.past_sequence_lengths = shared_banks_.past_sequence_lengths->narrow({{0, 0, n_req}});
    input.total_sequence_lengths = shared_banks_.total_sequence_lengths->narrow({{0, 0, n_req}});
    input.input_offsets = shared_banks_.input_offsets->narrow({{0, 0, n_req + 1}});
    input.cu_seqlens = shared_banks_.cu_seqlens->narrow({{0, 0, n_req + 1}});
    input.slot_mapping = shared_banks_.slot_mapping->narrow({{0, 0, bucket}});
    set_zeros(input.input_ids.value());
    set_zeros(input.past_sequence_lengths.value());
    set_zeros(input.total_sequence_lengths.value());

    std::vector<int64_t> position_ids_vec(bucket);
    position_ids_vec[0] = static_cast<int64_t>(decode_past);
    for (size_t i = 0; i < mid_q; ++i) {
        position_ids_vec[decode_q + i] = static_cast<int64_t>(mid_past + i);
    }
    infinicore::context::memcpyH2D(
        input.position_ids.value()->data(), position_ids_vec.data(), bucket * sizeof(int64_t), false);

    const std::vector<int32_t> past_lengths_vec{
        static_cast<int32_t>(decode_past), static_cast<int32_t>(mid_past)};
    const std::vector<int32_t> total_lengths_vec{
        static_cast<int32_t>(decode_total), static_cast<int32_t>(mid_total)};
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

    const std::vector<int32_t> input_offsets_vec{
        0, static_cast<int32_t>(decode_q), static_cast<int32_t>(bucket)};
    // cu_seqlens = cumulative KV lengths (matches processor MIXED packing).
    const std::vector<int32_t> cu_seqlens_vec{
        0, static_cast<int32_t>(decode_total), static_cast<int32_t>(decode_total + mid_total)};
    infinicore::context::memcpyH2D(
        input.input_offsets.value()->data(),
        input_offsets_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);
    infinicore::context::memcpyH2D(
        input.cu_seqlens.value()->data(),
        cu_seqlens_vec.data(),
        (n_req + 1) * sizeof(int32_t),
        false);

    const size_t block_per_req = nblocks;
    input.block_tables = block_tables_holder_->as_strided({n_req, block_per_req}, {(ptrdiff_t)block_per_req, 1});
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const size_t block_size = paged_config != nullptr ? paged_config->block_size() : 256;
    const size_t spans[2] = {decode_total, mid_total};
    size_t row_block_cursor = 0;
    for (size_t row = 0; row < n_req; ++row) {
        std::vector<int32_t> block_row(block_per_req, -1);
        const size_t blocks_needed = (spans[row] + block_size - 1) / block_size;
        for (size_t b = 0; b < blocks_needed && b < block_per_req; ++b) {
            block_row[b] = static_cast<int32_t>(row_block_cursor + b);
        }
        row_block_cursor += blocks_needed;
        auto row_tensor = input.block_tables.value()->narrow({{0, row, 1}});
        infinicore::context::memcpyH2D(
            row_tensor->data(), block_row.data(), block_per_req * sizeof(int32_t), false);
    }

    std::vector<int64_t> slot_mapping_vec(bucket);
    slot_mapping_vec[0] = static_cast<int64_t>(decode_past);
    for (size_t i = 0; i < mid_q; ++i) {
        slot_mapping_vec[decode_q + i] = static_cast<int64_t>(mid_past + i);
    }
    infinicore::context::memcpyH2D(
        input.slot_mapping.value()->data(), slot_mapping_vec.data(), bucket * sizeof(int64_t), false);

    // Row0 decode (lm_head + sample); row1 continuing mid (no sample).
    input.is_final_prefill_chunk = {true, false};
    return input;
}

void PiecewisePrefillCompiler::capture_bucket_(size_t bucket, bool mid_chunk_capture) {
    const auto rank_device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    infinicore::context::setDevice(rank_device);
    // Prefill: FA host-break + AOT MoE under full_and_piecewise.
    infinicore::context::InferencePhaseGuard phase_guard(
        infinicore::context::InferencePhase::Prefill);

    auto &piecewise_flag = infinilm::global_state::get_forward_context().piecewise;
    struct CaptureGuard {
        infinilm::global_state::PiecewisePrefillState &pw;
        explicit CaptureGuard(infinilm::global_state::PiecewisePrefillState &p) : pw(p) { pw.compile_capture_active = true; }
        ~CaptureGuard() { pw.compile_capture_active = false; }
    } capture_guard(piecewise_flag);

    const size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
    const size_t num_layers = model_->native_piecewise_num_layers();
    bind_bucket_staging_(bucket, num_layers);
    auto bucket_input = make_bucket_input_(bucket, nblocks, max_capture_req_, mid_chunk_capture);
    set_attn_metadata(bucket_input);

    BucketGraphs graphs;
    graphs.input = std::move(bucket_input);

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = bucket;
    // Dry-run + QKV CG must use the C++ pre path (not inductor). Inductor AOT does not
    // populate MiniCPM gate_score_cache_; allocating that buffer inside startGraphRecording
    // corrupts the QKV-only graph on MetaX.
    piecewise.allow_inductor_pre_attn = false;
    piecewise.phase = global_state::PiecewiseCapturePhase::None;

    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;

    size_t capture_layers = num_layers;
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_MAX_LAYERS")) {
        capture_layers = std::min(num_layers, static_cast<size_t>(std::stoul(raw)));
    }

    graphs.logits_holder = shared_banks_.logits_holder->narrow({{1, 0, bucket}});

    // Eager warmup dry-run before capture (NONE-equivalent).
    // Dry-run / capture use valid_seq_len==bucket (exact). Runtime mid-chunk
    // with seq_len==graph_bucket shares this staging; pad-up CG replays the
    // same banks with valid_seq_len=L + zeroed tails.
    // Dry-run must allocate MiniCPM gate buffers before CG (outside recording).
    const bool qkv_only = pre_attn_qkv_only();
    model_->native_piecewise_embed(graphs.input, hidden);
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        if (qkv_only) {
            model_->native_piecewise_pre_attn_qkv_layer(layer, graphs.input, hidden, residual);
            model_->native_piecewise_pre_attn_rope_layer(layer, graphs.input);
        } else {
            model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        }
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait();
        }
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait();
        }
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait();
        }
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
        }
    }
    // Mid chunks do not sample; skip lm_head dry-run/capture for mid key.
    if (!mid_chunk_capture) {
        model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    }
    // Reset KV so capture FA sees the same empty-cache shape as dry-run start.
    // Stale KV + many cudaGraphExecs on QY previously LOOR'd mid-ladder FA.
    {
        auto &kv_cache_vec = infinilm::global_state::get_forward_context().kv_cache_vec;
        for (auto &kv : kv_cache_vec) {
            if (kv) {
                set_zeros(kv);
            }
        }
        infinicore::context::syncDevice();
    }
    graphs.pre_attn.resize(capture_layers);
    graphs.post_attn.resize(capture_layers);

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(graphs.input, hidden);

    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;

        barrier_->wait();
        // Product: one pre piece [CG: LN+QKV+RoPE]. Kill-switch QKV_ONLY restores
        // Phase-2c QKV CG + eager RoPE. Inductor AOT packages bake RoPE — keep
        // allow_inductor_pre_attn=false so C++ path enters the shape-keyed CG.
        piecewise.allow_inductor_pre_attn = false;
        if (qkv_only) {
            infinicore::context::startGraphRecording();
            model_->native_piecewise_pre_attn_qkv_layer(layer, graphs.input, hidden, residual);
            graphs.pre_attn[layer] = infinicore::context::stopGraphRecording();
            infinicore::context::syncStream();
            model_->native_piecewise_pre_attn_rope_layer(layer, graphs.input);
            infinicore::context::syncStream();
        } else {
            infinicore::context::startGraphRecording();
            model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
            graphs.pre_attn[layer] = infinicore::context::stopGraphRecording();
            infinicore::context::syncStream();
        }

        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        barrier_->wait();
        infinicore::context::startGraphRecording();
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        graphs.post_attn[layer] = infinicore::context::stopGraphRecording();
        assert_post_one_device_seg_("piecewise_prefill_post", layer, graphs.post_attn[layer]);
        // Drain before next layer: QY GraphInstantiate failures otherwise sticky-LOOR FA.
        infinicore::context::syncDevice();
        barrier_->wait();
    }

    if (!mid_chunk_capture) {
        piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
        barrier_->wait();
        infinicore::context::startGraphRecording();
        model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
        graphs.lm_head = infinicore::context::stopGraphRecording();
        barrier_->wait();
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    graphs.hidden_states = piecewise.hidden_states;
    graphs.residual = piecewise.residual;
    graphs.ar_staging = piecewise.ar_staging;
    graphs.layer_staging = piecewise.layer_staging;
    if (mid_chunk_capture) {
        compiled_mid_[bucket] = std::move(graphs);
    } else {
        compiled_[bucket] = std::move(graphs);
    }
    const size_t captured_segments =
        capture_layers * 2 + (mid_chunk_capture ? 0 : 1);
    spdlog::info(
        "native piecewise CG: captured bucket={} mid={} layers={} segments={} pre={}",
        bucket,
        mid_chunk_capture,
        capture_layers,
        captured_segments,
        qkv_only ? "qkv_only" : "qkv_rope");
}

void PiecewisePrefillCompiler::capture_mixed_mid_bucket_(size_t bucket) {
    const auto rank_device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    infinicore::context::setDevice(rank_device);
    infinicore::context::InferencePhaseGuard phase_guard(
        infinicore::context::InferencePhase::Prefill);

    auto &piecewise_flag = infinilm::global_state::get_forward_context().piecewise;
    struct CaptureGuard {
        infinilm::global_state::PiecewisePrefillState &pw;
        explicit CaptureGuard(infinilm::global_state::PiecewisePrefillState &p) : pw(p) {
            pw.compile_capture_active = true;
        }
        ~CaptureGuard() { pw.compile_capture_active = false; }
    } capture_guard(piecewise_flag);

    const size_t nblocks =
        dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
    const size_t num_layers = model_->native_piecewise_num_layers();
    bind_bucket_staging_(bucket, num_layers);
    auto bucket_input = make_mixed_mid_bucket_input_(bucket, nblocks);
    set_attn_metadata(bucket_input);

    BucketGraphs graphs;
    graphs.input = std::move(bucket_input);

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = bucket;
    // MIXED mid dual-capture: never bake inductor-in-CG (same as homo mid).
    piecewise.allow_inductor_pre_attn = false;
    piecewise.phase = global_state::PiecewiseCapturePhase::None;

    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;

    size_t capture_layers = num_layers;
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_MAX_LAYERS")) {
        capture_layers = std::min(num_layers, static_cast<size_t>(std::stoul(raw)));
    }

    graphs.logits_holder = shared_banks_.logits_holder->narrow({{1, 0, bucket}});

    // Eager warmup dry-run (includes lm_head — decode row needs logits).
    model_->native_piecewise_embed(graphs.input, hidden);
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait();
        }
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait();
        }
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
            barrier_->wait();
        }
        if (infinilm::global_state::piecewise_inductor_segment_enabled()) {
            infinicore::context::syncDevice();
        }
    }
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    // Shape-keyed QKV comes from compiled_[bucket]; do not store MIXED pre_attn.
    graphs.pre_attn.resize(capture_layers);
    graphs.post_attn.resize(capture_layers);

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(graphs.input, hidden);

    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;

        barrier_->wait();
        piecewise.allow_inductor_pre_attn = false;
        // Eager QKV+RoPE for capture continuity only — prefer path uses final-bucket QKV CG.
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        infinicore::context::syncStream();

        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        barrier_->wait();
        infinicore::context::startGraphRecording();
        model_->native_piecewise_post_attn_cg_layer(layer, graphs.input, hidden, residual);
        graphs.post_attn[layer] = infinicore::context::stopGraphRecording();
        assert_post_one_device_seg_("piecewise_prefill_post", layer, graphs.post_attn[layer]);
        barrier_->wait();
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
    barrier_->wait();
    infinicore::context::startGraphRecording();
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    graphs.lm_head = infinicore::context::stopGraphRecording();
    barrier_->wait();

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    graphs.hidden_states = piecewise.hidden_states;
    graphs.residual = piecewise.residual;
    graphs.ar_staging = piecewise.ar_staging;
    graphs.layer_staging = piecewise.layer_staging;
    compiled_mixed_mid_[bucket] = std::move(graphs);
    // post per layer + lm_head (no MIXED pre CG).
    const size_t captured_segments = capture_layers + 1;
    spdlog::info(
        "native piecewise CG: captured bucket={} mixed_mid=1 n_req={} layers={} segments={} "
        "pre=skip_use_final_qkv",
        bucket,
        mixed_mid_capture_req_,
        capture_layers,
        captured_segments);
}

void PiecewisePrefillCompiler::warmup_inductor_segments_(size_t nblocks, size_t n_req) {
    if (!infinilm::global_state::piecewise_inductor_segment_enabled()) {
        return;
    }
    const size_t num_layers = model_->native_piecewise_num_layers();
    if (num_layers == 0 || capture_buckets_.empty()) {
        return;
    }
    size_t warmed_buckets = 0;
    for (size_t bucket : capture_buckets_) {
        if (!infinilm::global_state::bucket_is_inductor_eligible(bucket)) {
            continue;
        }
        bind_bucket_staging_(bucket, num_layers);
        auto bucket_input = make_bucket_input_(bucket, nblocks, n_req);
        const auto &positions = bucket_input.position_ids.value();
        auto positions_padded = infinicore::Tensor::zeros(
            {1, bucket}, infinicore::DataType::kInt64, infinicore::context::getDevice());
        auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
        for (size_t layer = 0; layer < num_layers; ++layer) {
            if (!infinicore::op::inductor_segment_impl::has_package(
                    infinicore::op::PiecewiseInductorSegmentId::PreAttn, layer, bucket)) {
                continue;
            }
            /* inductor warmup stubbed */;
            infinicore::context::syncDevice();
        }
        ++warmed_buckets;
    }
    if (warmed_buckets > 0) {
        auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
        piecewise.layer_staging.clear();
        piecewise.hidden_states = infinicore::Tensor();
        piecewise.residual = infinicore::Tensor();
        piecewise.ar_staging = infinicore::Tensor();
        infinicore::context::setDevice(
            infinilm::global_state::get_tensor_model_parallel_rank_info().device);
        infinicore::context::syncDevice();
        spdlog::info(
            "piecewise inductor: eager AOT warmup buckets={} (per-rank, before CG capture)",
            warmed_buckets);
    }
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
    // Capture width ≥ MAX_BATCH_SIZE so mixed/multi-req PIECEWISE can replay with
    // runtime_n_req ≤ compiled_n_req (vLLM mixed_mode=PIECEWISE). Opt-in override:
    // INFINI_MAX_PREFILL_BATCH. Default 8 (Band C). Set to 1 only for Gate-D bisect.
    max_capture_req_ = 8;
    if (const char *raw = std::getenv("INFINI_MAX_PREFILL_BATCH")) {
        max_capture_req_ = std::max<size_t>(1, std::stoul(raw));
    } else if (const char *raw = std::getenv("MAX_BATCH_SIZE")) {
        max_capture_req_ = std::max<size_t>(1, std::stoul(raw));
    }
    size_t max_bucket = capture_buckets_.empty() ? 0 : capture_buckets_.front();
    block_tables_holder_ = infinicore::Tensor::empty(
        {max_capture_req_ * nblocks}, infinicore::DataType::kInt32, infinicore::context::getDevice());
    set_zeros(block_tables_holder_);
    const size_t num_layers = model_->native_piecewise_num_layers();
    const size_t free_before = device_free_bytes_();
    // M2: allocate shared banks before inductor warmup so warmup also uses prefix views.
    allocate_shared_banks_(max_bucket, num_layers, max_capture_req_);
    spdlog::info(
        "native piecewise CG: capture warmup n_req={} (metadata only, hidden [1,bucket] shared max={})",
        max_capture_req_,
        max_bucket);

    const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
    const int tp_size = infinilm::global_state::get_tensor_model_parallel_world_size();
    if (tp_size > 1) {
        barrier_->wait();
    }
    verify_inductor_packages_(model_, capture_buckets_);
    warmup_inductor_segments_(nblocks, max_capture_req_);
    if (tp_size > 1) {
        barrier_->wait();
    }

    compiled_.clear();
    compiled_mid_.clear();
    compiled_mixed_mid_.clear();
    mixed_mid_capture_req_ = std::min<size_t>(2, max_capture_req_);
    for (size_t bucket : capture_buckets_) {
        capture_bucket_(bucket, /*mid_chunk_capture=*/false);
        infinicore::context::syncDevice();
        // Dual capture: exact chunk-sized bucket also gets homo mid keys.
        // MIXED dual-capture (compiled_mixed_mid_) is opt-in via KEEP_MIXED_MID_CAPTURE;
        // product MIXED prefer replays final compiled_ banks (vLLM-like).
        // QY: SKIP_MID_CAPTURE avoids a second full layer ladder of cudaGraphExec.
        if (!skip_homo_mid_capture() && bucket == prefill_chunk_size_ && prefill_chunk_size_ > 0) {
            capture_bucket_(bucket, /*mid_chunk_capture=*/true);
            infinicore::context::syncDevice();
            if (keep_mixed_mid_capture() && mixed_mid_capture_req_ >= 2 && bucket >= 2) {
                capture_mixed_mid_bucket_(bucket);
                infinicore::context::syncDevice();
            }
        }
    }
    infinicore::context::syncDevice();
    const size_t free_after = device_free_bytes_();
    // Physical staging/I-O = shared max bank (O(max)); arenas = MoE scratch via
    // shared Python workspace after M3 (still summed per Graph CaptureArena).
    const size_t staging_bytes = shared_banks_.staging_physical_bytes;
    const size_t io_bytes =
        shared_banks_.io_physical_bytes + tensor_nbytes_(block_tables_holder_);
    size_t arena_bytes = 0;
    for (const auto &kv : compiled_) {
        arena_bytes += bucket_arena_nbytes_(kv.second);
    }
    for (const auto &kv : compiled_mid_) {
        arena_bytes += bucket_arena_nbytes_(kv.second);
    }
    for (const auto &kv : compiled_mixed_mid_) {
        arena_bytes += bucket_arena_nbytes_(kv.second);
    }
    const size_t staging_io_bytes = staging_bytes + io_bytes;
    const size_t free_delta =
        free_before >= free_after ? free_before - free_after : 0;
    std::ostringstream oss;
    for (size_t i = 0; i < capture_buckets_.size(); ++i) {
        if (i > 0) {
            oss << ',';
        }
        oss << capture_buckets_[i];
    }
    spdlog::info(
        "native piecewise CG: capture_buckets=[{}] max_seq={} chunk_size={} mid_keys={} "
        "mixed_mid_keys={}",
        oss.str(),
        max_seq_len_,
        prefill_chunk_size_,
        compiled_mid_.size(),
        compiled_mixed_mid_.size());
    // M1: greppable one-line CG VRAM split (before /health). Staging is physical O(max) after M2.
    spdlog::info(
        "cg_mem_budget tag=piecewise_prefill buckets={} mid_buckets={} mixed_mid_buckets={} "
        "staging_bytes={} io_bytes={} "
        "staging_io_bytes={} arena_bytes={} free_delta_bytes={} "
        "staging_GiB={:.3f} io_GiB={:.3f} staging_io_GiB={:.3f} arena_GiB={:.3f} "
        "free_delta_GiB={:.3f}",
        compiled_.size(),
        compiled_mid_.size(),
        compiled_mixed_mid_.size(),
        staging_bytes,
        io_bytes,
        staging_io_bytes,
        arena_bytes,
        free_delta,
        bytes_to_gib_(staging_bytes),
        bytes_to_gib_(io_bytes),
        bytes_to_gib_(staging_io_bytes),
        bytes_to_gib_(arena_bytes),
        bytes_to_gib_(free_delta));
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
        if (runtime_n_req < compiled_n_req) {
            auto past_tail = graph_input.past_sequence_lengths.value()->narrow(
                {{0, runtime_n_req, compiled_n_req - runtime_n_req}});
            set_zeros(past_tail);
        }
    }
    graph_input.total_sequence_lengths.value()
        ->narrow({{0, 0, runtime_n_req}})
        ->copy_from(runtime.total_sequence_lengths.value());
    if (runtime_n_req < compiled_n_req) {
        auto total_tail = graph_input.total_sequence_lengths.value()->narrow(
            {{0, runtime_n_req, compiled_n_req - runtime_n_req}});
        set_zeros(total_tail);
    }
    graph_input.input_offsets.value()
        ->narrow({{0, 0, runtime_n_req + 1}})
        ->copy_from(runtime.input_offsets.value());
    graph_input.cu_seqlens.value()
        ->narrow({{0, 0, runtime_n_req + 1}})
        ->copy_from(runtime.cu_seqlens.value());
    // Pad unused capture-width rows with the last cumulative value (not zeros).
    // Zero-fill would make cu_seqlens/input_offsets non-monotonic
    // (…, total_tokens, 0, …) → negative FA varlen spans if anything reads
    // compiled_n_req width.
    if (runtime_n_req < compiled_n_req) {
        const size_t off_tail = compiled_n_req - runtime_n_req;
        auto last_off = graph_input.input_offsets.value()
                            ->narrow({{0, runtime_n_req, 1}})
                            ->to(infinicore::Device{infinicore::Device::Type::kCpu});
        const int32_t last_val =
            *reinterpret_cast<const int32_t *>(last_off->data());
        std::vector<int32_t> pad(off_tail, last_val);
        auto offsets_tail = graph_input.input_offsets.value()->narrow(
            {{0, runtime_n_req + 1, off_tail}});
        infinicore::context::memcpyH2D(
            offsets_tail->data(), pad.data(), pad.size() * sizeof(int32_t), false);
        auto cu_tail = graph_input.cu_seqlens.value()->narrow(
            {{0, runtime_n_req + 1, off_tail}});
        infinicore::context::memcpyH2D(
            cu_tail->data(), pad.data(), pad.size() * sizeof(int32_t), false);
    }

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
        // Prefer decode-style pad hygiene: zero slot tails (not -1). Prefill FA is
        // eager, but post CG may still touch padded width; -1 slots ATU under
        // rearrange when pad-up CG is on.
        auto slot_tail = graph_input.slot_mapping.value()->narrow({{0, valid_seq_len, bucket - valid_seq_len}});
        set_zeros(slot_tail);
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
    // Phase-scoped MoE: Prefill must not take Triton-in-graph path.
    // Under full_and_piecewise MoE is host-break in Prefill (Decode-only adaptive).
    infinicore::context::InferencePhaseGuard phase_guard(
        infinicore::context::InferencePhase::Prefill);
    last_prefill_executed_ = false;
    const bool profile = rank_worker_profile_enabled();
    const double t_total0 = profile ? monotonic_ms() : 0.0;
    const size_t seq_len = compute_prefill_len(input);
    const size_t padded = padded_bucket_for(seq_len);
    const size_t graph_bucket = graph_replay_bucket_for_padded(padded);
    const size_t runtime_n_req = input.block_tables.has_value() ? input.block_tables.value()->size(0) : 1;
    // Decode rows set is_final=True for sampling; that must not hide mid-prefill
    // rows (MIXED). mid_chunk ← any False flag; need_lm_head ← any True / empty.
    const bool mid_chunk =
        InfinilmModel::has_nonfinal_prefill_chunk(input.is_final_prefill_chunk);
    const bool need_lm_head =
        InfinilmModel::any_final_prefill_chunk(input.is_final_prefill_chunk);
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise run_prefill begin seq_len={} padded={} graph_bucket={} "
            "n_req={} mid_chunk={} need_lm_head={}",
            seq_len,
            padded,
            graph_bucket,
            runtime_n_req,
            mid_chunk,
            need_lm_head);
    }
    if (compiled_.find(graph_bucket) == compiled_.end()) {
        ++prefill_misses_;
        return std::nullopt;
    }
    if (runtime_n_req > max_capture_req_) {
        ++prefill_misses_;
        spdlog::debug(
            "piecewise run_prefill: runtime_n_req={} > max_capture_req={} → eager",
            runtime_n_req,
            max_capture_req_);
        return std::nullopt;
    }
    const bool inductor_mode = infinilm::global_state::piecewise_inductor_segment_enabled();
    // Pad-up (seq_len < graph_bucket): product default replays token-bucket CG with
    // valid_seq_len=L + zeroed tails (INFINI_PIECEWISE_PAD_UP_CG, default ON).
    // Kill-switch =0 restores legacy eager-only pad-up (PlannedMeta / SIGSEGV history).
    const bool pad_up = seq_len != graph_bucket;
    const bool pad_up_cg_on = piecewise_pad_up_cg_enabled();
    const bool pad_up_blocks_cg = pad_up && !pad_up_cg_on;
    const bool qkv_only = pre_attn_qkv_only();

    const bool have_mid_graphs = [&]() {
        auto it = compiled_mid_.find(graph_bucket);
        if (it == compiled_mid_.end()) {
            return false;
        }
        return !it->second.pre_attn.empty() && it->second.pre_attn[0]
            && !it->second.post_attn.empty() && it->second.post_attn[0];
    }();
    const bool have_mixed_mid_graphs = [&]() {
        auto it = compiled_mixed_mid_.find(graph_bucket);
        if (it == compiled_mixed_mid_.end()) {
            return false;
        }
        // Legacy KEEP_MIXED_MID_CAPTURE banks: post + lm_head only.
        return !it->second.post_attn.empty() && it->second.post_attn[0]
            && static_cast<bool>(it->second.lm_head);
    }();
    (void)have_mixed_mid_graphs; // telemetry / legacy banks only; prefer uses final compiled_.
    // Runtime past of request 0 (representative). Continuing homo mids (past>0) use
    // compiled_mid_; first mid (past==0) uses shape-keyed final compiled_ pre+post
    // (fused LN+QKV+RoPE CG by default; QKV_ONLY → eager RoPE).
    int32_t runtime_past0 = 0;
    if (input.past_sequence_lengths.has_value()
        && input.past_sequence_lengths.value()->size(0) > 0) {
        auto cpu = input.past_sequence_lengths.value()->to(infinicore::Device{infinicore::Device::Type::kCpu});
        runtime_past0 = reinterpret_cast<const int32_t *>(cpu->data())[0];
    }
    const bool past_matches_mid_capture = runtime_past0 > 0;
    // Shape-keyed QKV + post live on compiled_[bucket] (final capture).
    const bool have_shape_qkv =
        compiled_.count(graph_bucket) > 0
        && !compiled_.at(graph_bucket).pre_attn.empty()
        && compiled_.at(graph_bucket).pre_attn[0];
    const bool have_final_post =
        compiled_.count(graph_bucket) > 0
        && !compiled_.at(graph_bucket).post_attn.empty()
        && compiled_.at(graph_bucket).post_attn[0];
    // Telemetry only: former layout gate 1+(bucket-1); prefer no longer requires it.
    bool mixed_layout_matches_capture = false;
    int32_t mixed_q0 = -1;
    int32_t mixed_q1 = -1;
    int32_t runtime_mid_past = runtime_past0;
    if (runtime_n_req >= 2 && input.input_offsets.has_value()
        && input.input_offsets.value()->size(0) >= runtime_n_req + 1) {
        auto off_cpu = input.input_offsets.value()->to(infinicore::Device{infinicore::Device::Type::kCpu});
        const auto *offs = reinterpret_cast<const int32_t *>(off_cpu->data());
        mixed_q0 = offs[1] - offs[0];
        mixed_q1 = offs[2] - offs[1];
        const int32_t expect_mid_q = static_cast<int32_t>(graph_bucket) - 1;
        mixed_layout_matches_capture = (mixed_q0 == 1 && mixed_q1 == expect_mid_q);
        if (input.past_sequence_lengths.has_value()
            && input.past_sequence_lengths.value()->size(0) >= 2) {
            auto past_cpu = input.past_sequence_lengths.value()->to(infinicore::Device{infinicore::Device::Type::kCpu});
            // Row1 past under decode-first packs (telemetry).
            runtime_mid_past = reinterpret_cast<const int32_t *>(past_cpu->data())[1];
        }
    }
    // vLLM-like: MIXED exact-width (or pad-up CG) PIECEWISE uses final token-bucket
    // graphs (compiled_[bucket]). ENABLE/REPRO is the product kill-switch.
    const bool prefer_mixed_piecewise =
        mid_chunk && need_lm_head && !pad_up_blocks_cg && have_shape_qkv && have_final_post
        && (enable_mixed_mid_cg() || repro_mixed_mid_cg());
    // Alias for greppability with prior probe strings.
    const bool prefer_mixed_mid_graphs = prefer_mixed_piecewise;
    // Product: exact homogeneous mid + past>0 → mid dual-capture graphs.
    // Part A bisect: ALLOW_MIDCHUNK_CG forces final graphs on mid.
    const bool prefer_mid_graphs =
        !prefer_mixed_piecewise
        && mid_chunk && !need_lm_head && !pad_up_blocks_cg && have_mid_graphs
        && past_matches_mid_capture && !repro_allow_midchunk_cg();
    // Product: homogeneous first mid (past==0) → final shape-keyed pre+post (not
    // compiled_mid_, which was captured with past=chunk_size). Do not route onto mid banks.
    const bool prefer_past0_mid_final_graphs =
        !prefer_mixed_piecewise
        && mid_chunk && !need_lm_head && !pad_up_blocks_cg && have_shape_qkv
        && runtime_past0 == 0;
    // bucket_graphs: prefer_mid → compiled_mid_; else compiled_ (mixed + past0 + final).
    auto &bucket_graphs = prefer_mid_graphs
        ? compiled_mid_.at(graph_bucket)
        : compiled_.at(graph_bucket);

    // Mid CG ok: past>0 mid / MIXED final-bank / past0→final, or repro lifts onto final graphs.
    const bool mid_cg_ok =
        prefer_mid_graphs
        || prefer_mixed_piecewise
        || prefer_past0_mid_final_graphs
        || (mid_chunk && !pad_up_blocks_cg
            && (repro_allow_midchunk_cg() || repro_skip_midchunk_eager()));
    const bool force_mid_eager = mid_chunk && !mid_cg_ok;
    // Step-level eager hist: pad_up_eager > mid_chunk > missing > exact_cg.
    // Successful pad-up CG counts as exact_cg (not pw_eager_pad_up).
    {
        const bool missing_graph =
            (!have_shape_qkv
             && (bucket_graphs.pre_attn.empty() || !bucket_graphs.pre_attn[0]))
            || bucket_graphs.post_attn.empty() || !bucket_graphs.post_attn[0];
        const bool use_pad_up_cg =
            pad_up && pad_up_cg_on && !missing_graph && !force_mid_eager;
        const bool pad_up_eager = pad_up && !use_pad_up_cg;
        const bool mid_chunk_eager = force_mid_eager && !pad_up_eager;
        dispatch_hist::record_piecewise_eager(pad_up_eager, mid_chunk_eager, missing_graph);
    }
    const double t_copy0 = profile ? monotonic_ms() : 0.0;
    copy_runtime_into_bucket_(bucket_graphs, input, seq_len);
    // Keep sample/lm_head row flags aligned with the live MIXED pack (capture baked
    // decode-first {true,false}; runtime must match for eager lm_head gather).
    if (!input.is_final_prefill_chunk.empty()) {
        bucket_graphs.input.is_final_prefill_chunk = input.is_final_prefill_chunk;
    }
    set_attn_metadata_for_varlen_batch(bucket_graphs.input, input);
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
    if (!mid_chunk) {
        // Exact-width and pad-up final chunks both use inductor when enabled *and*
        // not taking native pad-up CG (kill-switch / miss falls back to eager AOT).
        // Pad-up still sets piecewise.valid_seq_len=L; AOT pads positions/hidden to bucket.
        piecewise.allow_inductor_pre_attn =
            inductor_mode && !repro_skip_final_inductor()
            && (pad_up_blocks_cg || !pad_up);
    } else if (prefer_past0_mid_final_graphs || prefer_mixed_piecewise || prefer_mid_graphs) {
        // Product mid CG prefers: fused pre CG (or QKV_ONLY+eager RoPE) — never
        // fused PreAttn AOTI (packages bake RoPE; incompatible with host-break RoPE).
        piecewise.allow_inductor_pre_attn = false;
    } else if (repro_skip_midchunk_eager()) {
        // A2: stress inductor-in-CG mid (out of product default).
        piecewise.allow_inductor_pre_attn = inductor_mode;
    } else {
        // Remaining mid paths: keep inductor out.
        piecewise.allow_inductor_pre_attn = false;
    }

    // Fresh residual each replay (matches capture warmup); hidden prefix comes from embed.
    set_zeros(piecewise.residual);
    // Pad-up: clear stale CG tails on staging/hidden/logits before embed so pad
    // positions do not pollute layernorm / MoE (RC-4).
    if (seq_len < graph_bucket) {
        clear_stale_bucket_tails_(piecewise, bucket_graphs.logits_holder, seq_len, graph_bucket);
    }

    model_->native_piecewise_embed(bucket_graphs.input, piecewise.hidden_states);


    // Always walk the full model; layers without a captured graph run eager
    // (QY may cap capture via INFINI_NATIVE_CG_MAX_LAYERS to fit cudaGraphExec budget).
    const size_t num_layers = model_->native_piecewise_num_layers();
    const size_t captured_post = bucket_graphs.post_attn.size();
    const size_t captured_pre = bucket_graphs.pre_attn.size();
    (void)captured_pre;
    // Mid with dual mid graphs: CG replay; inductor-in-CG layers stay eager unless A2.
    // Final-chunk inductor pre-attn needs the same eager post replay (B4 tail).
    // Pad-up: CG when PAD_UP_CG on; legacy kill-switch forces eager post/lm_head.
    // MIXED prefer: pre CG + eager attn + post CG from compiled_; lm_head always eager
    // when mid+need_lm_head (live is_final_prefill_chunk; do not replay capture-baked gather).
    const bool use_eager_post =
        pad_up_blocks_cg
        || force_mid_eager
        || (inductor_mode && piecewise.allow_inductor_pre_attn);
    const bool use_eager_lm_head =
        pad_up_blocks_cg
        || force_mid_eager
        || (need_lm_head && mid_chunk)
        || prefer_mid_graphs
        || prefer_past0_mid_final_graphs
        || !bucket_graphs.lm_head
        || (inductor_mode && piecewise.allow_inductor_pre_attn);
    {
        // Telemetry: first past=0 mid, first past>0 homo mid, first MIXED mid prefer,
        // and first short final pad-up CG.
        static bool logged_mid_past0 = false;
        static bool logged_mid_past_gt0 = false;
        static bool logged_mixed_mid = false;
        static bool logged_pad_up_cg = false;
        if (pad_up && pad_up_cg_on && !force_mid_eager && !logged_pad_up_cg) {
            logged_pad_up_cg = true;
            spdlog::info(
                "pad_up_cg=true valid_seq_len={} graph_bucket={} mid_chunk={} "
                "pre_rope_in_cg={}",
                seq_len,
                graph_bucket,
                mid_chunk,
                !qkv_only);
        }
        if (mid_chunk && !pad_up_blocks_cg && graph_bucket >= 2048) {
            const bool do_log = (runtime_past0 <= 0 && !logged_mid_past0)
                || (runtime_past0 > 0 && !need_lm_head && !logged_mid_past_gt0)
                || (prefer_mixed_piecewise && !logged_mixed_mid)
                || (need_lm_head && !logged_mixed_mid);
            if (do_log) {
                if (runtime_past0 <= 0 && !need_lm_head) {
                    logged_mid_past0 = true;
                } else if (!need_lm_head) {
                    logged_mid_past_gt0 = true;
                }
                if (prefer_mixed_piecewise || need_lm_head) {
                    logged_mixed_mid = true;
                }
                const bool pre_cg =
                    have_shape_qkv && !pad_up_blocks_cg && !force_mid_eager;
                spdlog::info(
                    "mid_chunk_cg_probe: mid_chunk={} past_len={} mid_past={} seq_len={} "
                    "graph_bucket={} force_mid_eager={} prefer_mid_graphs={} "
                    "prefer_past0_mid_final_graphs={} prefer_mixed_piecewise={} "
                    "prefer_mixed_mid_graphs={} "
                    "mid_cg_ok={} past_matches_mid_capture={} "
                    "mixed_layout_matches_capture={} mixed_q0={} mixed_q1={} n_req={} "
                    "allow_inductor_pre_attn={} use_eager_post={} use_eager_lm_head={} "
                    "pre_qkv_cg={} pre_rope_in_cg={} pre_rope_eager={} "
                    "have_shape_qkv={} have_final_post={} pad_up={} pad_up_cg_on={}",
                    mid_chunk,
                    runtime_past0,
                    runtime_mid_past,
                    seq_len,
                    graph_bucket,
                    force_mid_eager,
                    prefer_mid_graphs,
                    prefer_past0_mid_final_graphs,
                    prefer_mixed_piecewise,
                    prefer_mixed_mid_graphs,
                    mid_cg_ok,
                    past_matches_mid_capture,
                    mixed_layout_matches_capture,
                    mixed_q0,
                    mixed_q1,
                    runtime_n_req,
                    piecewise.allow_inductor_pre_attn,
                    use_eager_post,
                    use_eager_lm_head,
                    pre_cg,
                    pre_cg && !qkv_only,
                    pre_cg && qkv_only,
                    have_shape_qkv,
                    have_final_post,
                    pad_up,
                    pad_up_cg_on);
            }
        }
    }
    double layers_ms = 0.0;
    const double t_layers0 = profile ? monotonic_ms() : 0.0;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        const double t_layer0 = profile ? monotonic_ms() : 0.0;
        barrier_->wait();
        const bool inductor_layer_mid =
            mid_chunk && layer_capture_inductor_pre_attn(layer, graph_bucket)
            && !repro_skip_midchunk_eager();
        // Product: fused pre CG (LN+QKV+RoPE). QKV_ONLY kill-switch: QKV CG + eager RoPE.
        // Prefer shape-keyed CG over inductor-in-pre (AOT packages bake RoPE).
        // Legacy pad-up kill-switch / force_mid_eager / mid-inductor stress → eager full pre.
        const bool force_eager_pre =
            pad_up_blocks_cg
            || force_mid_eager
            || inductor_layer_mid;

        // Prefer path pre: bucket_graphs is compiled_ for MIXED/past0/final, or
        // compiled_mid_ for homo continuing mid.
        std::shared_ptr<infinicore::graph::Graph> pre_graph;
        if (!force_eager_pre) {
            if (layer < bucket_graphs.pre_attn.size()
                && bucket_graphs.pre_attn[layer]) {
                pre_graph = bucket_graphs.pre_attn[layer];
            } else if (have_shape_qkv && layer < compiled_.at(graph_bucket).pre_attn.size()
                       && compiled_.at(graph_bucket).pre_attn[layer]) {
                pre_graph = compiled_.at(graph_bucket).pre_attn[layer];
            }
        }

        double pre_graph_run_ms = 0.0;
        double pre_eager_ms = 0.0;
        infinicore::graph::Graph::LastRunProfile pre_split{};
        if (pre_graph) {
            const double t_pre_g0 = profile ? monotonic_ms() : 0.0;
            pre_graph->run();
            pre_split = pre_graph->last_run_profile();
            // When Graph::run skips its tail sync, PROFILE_SYNC still needs a drain
            // for exclusive graph_run_ms; otherwise FA must see pre outputs.
            if (profile && rank_worker_profile_sync()) {
                if (pre_split.skipped_tail_sync || !pre_split.used_device) {
                    infinicore::context::syncDevice();
                }
            } else if (pre_split.skipped_tail_sync) {
                // FA is eager and reads QKV from pre CG — one stream drain.
                infinicore::context::syncStream();
            }
            if (profile) {
                pre_graph_run_ms = monotonic_ms() - t_pre_g0;
            }
            ++segment_replays_;
            if (qkv_only) {
                // Phase-2c: RoPE must see QKV staging writes; graph replay may be async.
                // Opt-in skip via INFINI_PIECEWISE_SKIP_QKV_ROPE_SYNC=1.
                if (!global_state::skip_qkv_rope_sync()) {
                    infinicore::context::syncDevice();
                }
                const double t_rope0 = profile ? monotonic_ms() : 0.0;
                model_->native_piecewise_pre_attn_rope_layer(layer, bucket_graphs.input);
                if (profile) {
                    pre_eager_ms = monotonic_ms() - t_rope0;
                }
            }
            // else: fused pre already includes RoPE — no host break / sync.
        } else {
            // Eager full pre (QKV+RoPE / inductor package when allowed).
            const double t_pre_e0 = profile ? monotonic_ms() : 0.0;
            model_->native_piecewise_pre_attn_layer(
                layer, bucket_graphs.input, piecewise.hidden_states, piecewise.residual);
            if (profile) {
                pre_eager_ms = monotonic_ms() - t_pre_e0;
            }
        }
        const double t_pre_attn = profile ? monotonic_ms() : 0.0;
        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, bucket_graphs.input);
        const double t_eager_attn = profile ? monotonic_ms() : 0.0;
        barrier_->wait();
        double post_graph_run_ms = 0.0;
        double post_eager_ms = 0.0;
        infinicore::graph::Graph::LastRunProfile post_split{};
        const bool have_post_graph =
            layer < captured_post && static_cast<bool>(bucket_graphs.post_attn[layer]);
        if (use_eager_post || inductor_layer_mid || !have_post_graph) {
            const double t_post_e0 = profile ? monotonic_ms() : 0.0;
            model_->native_piecewise_post_attn_cg_layer(
                layer, bucket_graphs.input, piecewise.hidden_states, piecewise.residual);
            barrier_->wait();
            if (profile) {
                post_eager_ms = monotonic_ms() - t_post_e0;
            }
        } else {
            const double t_post_g0 = profile ? monotonic_ms() : 0.0;
            bucket_graphs.post_attn[layer]->run();
            post_split = bucket_graphs.post_attn[layer]->last_run_profile();
            // Double-sync cut: Graph::run already synced unless SKIP_TAIL_SYNC.
            // PROFILE_SYNC exclusive timing: skip redundant syncDevice when Graph
            // already drained the stream (LIGHT_SYNC spirit / TP=1).
            if (profile && rank_worker_profile_sync()) {
                if (post_split.skipped_tail_sync || !post_split.used_device) {
                    infinicore::context::syncDevice();
                }
            }
            // Mid-layer: next consumer is another Graph::run / FA on same stream —
            // SKIP_TAIL_SYNC leaves work ordered; drain once before lm_head below.
            if (profile) {
                post_graph_run_ms = monotonic_ms() - t_post_g0;
            }
            ++segment_replays_;
            barrier_->wait();
        }
        if (profile) {
            // Barrier after Graph::run folds into post_attn_ms only; graph_run stays
            // Graph::run(+optional sync) exclusive.
            std::string extra;
            if (pre_graph) {
                append_graph_run_split(extra, "pre_graph", pre_split);
            }
            if (post_graph_run_ms > 0.0 || post_split.used_device ) {
                append_graph_run_split(extra, "post_graph", post_split);
            }
            spdlog::info(
                "rank_worker_profile: piecewise layer={} pre_attn_ms={:.3f} "
                "pre_graph_run_ms={:.3f} pre_eager_ms={:.3f} eager_attn_ms={:.3f} "
                "post_attn_ms={:.3f} post_graph_run_ms={:.3f} post_eager_ms={:.3f} "
                "layer_total_ms={:.3f}{}",
                layer,
                t_pre_attn - t_layer0,
                pre_graph_run_ms,
                pre_eager_ms,
                t_eager_attn - t_pre_attn,
                monotonic_ms() - t_eager_attn,
                post_graph_run_ms,
                post_eager_ms,
                monotonic_ms() - t_layer0,
                extra);
        }
    }
    if (profile) {
        layers_ms = monotonic_ms() - t_layers0;
    }
    if (need_lm_head) {
        const double t_lm0 = profile ? monotonic_ms() : 0.0;
        double lm_graph_run_ms = 0.0;
        double lm_eager_ms = 0.0;
        barrier_->wait();
        // Gate C: one stream drain before logits when post GraphExec stayed async.
        if (graph_skip_tail_sync_enabled()) {
            infinicore::context::syncStream();
        }
        if (use_eager_lm_head) {
            const double t_e0 = profile ? monotonic_ms() : 0.0;
            model_->native_piecewise_lm_head(
                bucket_graphs.input, piecewise.hidden_states, piecewise.residual, bucket_graphs.logits_holder);
            if (profile) {
                lm_eager_ms = monotonic_ms() - t_e0;
            }
        } else {
            const double t_g0 = profile ? monotonic_ms() : 0.0;
            bucket_graphs.lm_head->run();
            if (profile && rank_worker_profile_sync()) {
                const auto &lm_split = bucket_graphs.lm_head->last_run_profile();
                if (lm_split.skipped_tail_sync || !lm_split.used_device) {
                    infinicore::context::syncDevice();
                }
            }
            if (profile) {
                lm_graph_run_ms = monotonic_ms() - t_g0;
            }
            ++segment_replays_;
        }
        barrier_->wait();
        if (profile) {
            spdlog::info(
                "rank_worker_profile: piecewise lm_head_ms={:.3f} "
                "lm_head_graph_run_ms={:.3f} lm_head_eager_ms={:.3f}",
                monotonic_ms() - t_lm0,
                lm_graph_run_ms,
                lm_eager_ms);
        }
    } else if (graph_skip_tail_sync_enabled()) {
        // Mid-chunk without lm_head: still drain before returning (KV consumers).
        infinicore::context::syncStream();
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    ++prefill_hits_;
    last_prefill_executed_ = true;
    if (profile) {
        spdlog::info(
            "rank_worker_profile: piecewise run_prefill end layers_total_ms={:.3f} total_ms={:.3f} "
            "mid_chunk={} need_lm_head={} pad_up={}",
            layers_ms,
            monotonic_ms() - t_total0,
            mid_chunk,
            need_lm_head,
            pad_up);
    }
    if (global_state::ar_profile::enabled()
        && global_state::get_tensor_model_parallel_rank() == 0) {
        global_state::ar_profile::log_barrier_chunk_summary(
            "piecewise_replay", seq_len, runtime_n_req);
    }
    if (!need_lm_head) {
        return std::nullopt;
    }
    return bucket_graphs.logits_holder->narrow({{1, 0, seq_len}});
}

} // namespace infinilm::engine
