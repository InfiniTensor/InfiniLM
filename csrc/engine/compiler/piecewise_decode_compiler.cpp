#include "piecewise_decode_compiler.hpp"

#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "../compiled_prefill_flags.hpp"
#include "attn_metadata_utils.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace infinilm::engine {

namespace {

std::vector<size_t> parse_capture_batches_() {
    std::vector<size_t> batches;
    if (const char *raw = std::getenv("INFINI_DECODE_PIECEWISE_BATCHES")) {
        std::string spec(raw);
        size_t start = 0;
        while (start < spec.size()) {
            const size_t comma = spec.find(',', start);
            const std::string token =
                spec.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            if (!token.empty()) {
                batches.push_back(static_cast<size_t>(std::stoul(token)));
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
    }
    if (batches.empty()) {
        // G7 MC=1 is the critical path; widen via INFINI_DECODE_PIECEWISE_BATCHES.
        batches = {1};
    }
    std::sort(batches.begin(), batches.end());
    batches.erase(std::unique(batches.begin(), batches.end()), batches.end());
    return batches;
}

} // namespace

PiecewiseDecodeCompiler::PiecewiseDecodeCompiler(const std::shared_ptr<InfinilmModel> &model,
                                                 RankBarrier *barrier)
    : model_(model), barrier_(barrier) {
    enabled_ = native_piecewise_decode_enabled() && model_->supports_native_piecewise_prefill();
    if (!enabled_) {
        return;
    }
    capture_batches_ = parse_capture_batches_();
}

void PiecewiseDecodeCompiler::allocate_layer_staging_(size_t batch, size_t num_layers) {
    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.bucket_seq_len = batch;
    piecewise.layer_staging.clear();
    piecewise.layer_staging.resize(num_layers);
    const auto device = infinicore::context::getDevice();
    const auto &model_config = infinilm::global_state::get_infinilm_config().model_config;
    const auto dtype = model_config->get_dtype();
    const size_t hidden = model_config->get<size_t>("hidden_size");
    const size_t tp_size = std::max<size_t>(
        1, static_cast<size_t>(infinilm::global_state::get_tensor_model_parallel_world_size()));
    const size_t num_heads =
        model_config->get<size_t>("num_attention_heads") / static_cast<size_t>(tp_size);
    const size_t total_kv = model_config->get<size_t>("num_key_value_heads");
    const size_t num_kv_heads = total_kv < tp_size ? 1 : total_kv / tp_size;
    const size_t head_dim = model_config->get_head_dim();

    for (size_t i = 0; i < num_layers; ++i) {
        auto &st = piecewise.layer_staging[i];
        st.q_rope = infinicore::Tensor::empty({1, batch, num_heads, head_dim}, dtype, device);
        st.k_rope = infinicore::Tensor::empty({1, batch, num_kv_heads, head_dim}, dtype, device);
        st.v_rope = infinicore::Tensor::empty({1, batch, num_kv_heads, head_dim}, dtype, device);
        st.attn_output =
            infinicore::Tensor::empty({1, batch, num_heads * head_dim}, dtype, device);
    }
    piecewise.hidden_states = infinicore::Tensor::empty({1, batch, hidden}, dtype, device);
    piecewise.residual = infinicore::Tensor::empty({1, batch, hidden}, dtype, device);
    piecewise.ar_staging = infinicore::Tensor::empty({1, batch, hidden}, dtype, device);
}

InfinilmModel::Input PiecewiseDecodeCompiler::make_batch_input_(size_t batch,
                                                                size_t nblocks) const {
    InfinilmModel::Input input;
    const auto device = infinicore::context::getDevice();
    // Decode: tokens == requests (batch_size == input_width).
    input.input_ids = infinicore::Tensor::empty({1, batch}, infinicore::DataType::I64, device);
    input.position_ids = infinicore::Tensor::empty({batch}, infinicore::DataType::I64, device);
    input.total_sequence_lengths =
        infinicore::Tensor::empty({batch}, infinicore::DataType::I32, device);
    set_zeros(input.input_ids.value());
    set_zeros(input.position_ids.value());
    set_zeros(input.total_sequence_lengths.value());

    std::vector<int32_t> total_sequence_lengths_vec(batch, 1);
    infinicore::context::memcpyH2D(input.total_sequence_lengths.value()->data(),
                                   total_sequence_lengths_vec.data(),
                                   batch * sizeof(int32_t),
                                   false);

    input.input_offsets =
        infinicore::Tensor::empty({batch + 1}, infinicore::DataType::I32, device);
    std::vector<int32_t> input_offsets_vec(batch + 1, 0);
    std::iota(input_offsets_vec.begin(), input_offsets_vec.end(), 0);
    infinicore::context::memcpyH2D(input.input_offsets.value()->data(),
                                   input_offsets_vec.data(),
                                   (batch + 1) * sizeof(int32_t),
                                   false);

    input.cu_seqlens = infinicore::Tensor::empty({batch + 1}, infinicore::DataType::I32, device);
    infinicore::context::memcpyH2D(input.cu_seqlens.value()->data(),
                                   input_offsets_vec.data(),
                                   (batch + 1) * sizeof(int32_t),
                                   false);

    const size_t block_per_req = nblocks;
    input.block_tables =
        block_tables_holder_->as_strided({batch, block_per_req}, {(ptrdiff_t)block_per_req, 1});
    set_minus_one(input.block_tables.value());
    // Assign trivial block 0.. for capture warmup (overwritten on replay).
    for (size_t row = 0; row < batch; ++row) {
        std::vector<int32_t> block_row(block_per_req, -1);
        block_row[0] = static_cast<int32_t>(row);
        auto row_tensor = input.block_tables.value()->narrow({{0, row, 1}});
        infinicore::context::memcpyH2D(
            row_tensor->data(), block_row.data(), block_per_req * sizeof(int32_t), false);
    }

    input.slot_mapping = infinicore::Tensor::empty({batch}, infinicore::DataType::I64, device);
    set_zeros(input.slot_mapping.value());
    return input;
}

void PiecewiseDecodeCompiler::capture_batch_(size_t batch) {
    const auto rank_device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    infinicore::context::setDevice(rank_device);

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

    allocate_layer_staging_(batch, num_layers);
    auto batch_input = make_batch_input_(batch, nblocks);
    attn_metadata_utils::set_attn_metadata(batch_input);

    BatchGraphs graphs;
    graphs.input = std::move(batch_input);

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = batch;
    // Decode uses native gemm/norm CG segments; MoE AOTI is eager between runs.
    piecewise.allow_inductor_pre_attn = false;
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
    graphs.logits_holder =
        infinicore::Tensor::empty({1, batch, vocab_size}, dtype, infinicore::context::getDevice());

    // Eager dry-run before capture.
    model_->native_piecewise_embed(graphs.input, hidden);
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);
        model_->native_piecewise_post_attn_decode_cg_layer(layer, graphs.input, hidden, residual);
        model_->native_piecewise_eager_moe_layer(layer, graphs.input, hidden, residual);
    }
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);

    graphs.pre_attn.resize(capture_layers);
    graphs.post_attn.resize(capture_layers);

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(graphs.input, hidden);

    const bool need_barrier = barrier_ != nullptr && barrier_->num_ranks() > 1;
    size_t device_segs = 0;
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;

        if (need_barrier) {
            barrier_->wait("piecewise_decode_capture_pre_attn");
        }
        // Separate recordings: FA/MoE must NOT run under isGraphRecording on MetaX
        // (unified layer capture HTC-faults even with host_break splits).
        infinicore::context::startGraphRecording();
        model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
        graphs.pre_attn[layer] = infinicore::context::stopGraphRecording();
        if (graphs.pre_attn[layer] && graphs.pre_attn[layer]->has_device_exec()) {
            ++device_segs;
        }

        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        if (need_barrier) {
            barrier_->wait("piecewise_decode_capture_post_attn");
        }
        infinicore::context::startGraphRecording();
        model_->native_piecewise_post_attn_decode_cg_layer(layer, graphs.input, hidden, residual);
        graphs.post_attn[layer] = infinicore::context::stopGraphRecording();
        if (graphs.post_attn[layer] && graphs.post_attn[layer]->has_device_exec()) {
            ++device_segs;
        }

        model_->native_piecewise_eager_moe_layer(layer, graphs.input, hidden, residual);
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
    if (need_barrier) {
        barrier_->wait("piecewise_decode_capture_lm_head");
    }
    infinicore::context::startGraphRecording();
    model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
    graphs.lm_head = infinicore::context::stopGraphRecording();
    if (graphs.lm_head && graphs.lm_head->has_device_exec()) {
        ++device_segs;
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    graphs.hidden_states = piecewise.hidden_states;
    graphs.residual = piecewise.residual;
    graphs.ar_staging = piecewise.ar_staging;
    graphs.layer_staging = piecewise.layer_staging;
    graphs.device_segments = device_segs;
    device_segments_captured_ += device_segs;
    compiled_[batch] = std::move(graphs);

    spdlog::info(
        "native piecewise decode CG: captured batch={} layers={} device_segments={}",
        batch,
        capture_layers,
        device_segs);
}

void PiecewiseDecodeCompiler::compile() {
    if (!enabled_) {
        return;
    }
    if (model_->get_cache_config() == nullptr
        || !dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        spdlog::warn("piecewise decode CG: skipped (paged KV required)");
        enabled_ = false;
        return;
    }
    const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
    if (rank_info.tp_size > 1 && !decode_cg_tp_enabled()) {
        spdlog::info(
            "piecewise decode CG: skip capture (tp_size={} > 1; set INFINI_DECODE_CG_TP=1 to opt in)",
            rank_info.tp_size);
        enabled_ = false;
        return;
    }

    const size_t nblocks =
        dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
    const size_t max_batch = *std::max_element(capture_batches_.begin(), capture_batches_.end());
    block_tables_holder_ = infinicore::Tensor::empty(
        {nblocks * max_batch}, infinicore::DataType::I32, infinicore::context::getDevice());
    set_zeros(block_tables_holder_);

    spdlog::info("piecewise decode CG: compiling batches=[{}]", [&]() {
        std::ostringstream oss;
        for (size_t i = 0; i < capture_batches_.size(); ++i) {
            if (i) {
                oss << ',';
            }
            oss << capture_batches_[i];
        }
        return oss.str();
    }());

    for (size_t batch : capture_batches_) {
        capture_batch_(batch);
    }
}

void PiecewiseDecodeCompiler::copy_runtime_into_batch_(BatchGraphs &batch_graphs,
                                                       const InfinilmModel::Input &runtime) const {
    auto &graph_input = batch_graphs.input;
    graph_input.input_ids.value()->copy_from(runtime.input_ids.value());
    graph_input.position_ids.value()->copy_from(runtime.position_ids.value());
    graph_input.total_sequence_lengths.value()->copy_from(runtime.total_sequence_lengths.value());
    graph_input.input_offsets.value()->copy_from(runtime.input_offsets.value());
    graph_input.cu_seqlens.value()->copy_from(runtime.cu_seqlens.value());

    const size_t block_per_req = runtime.block_tables.value()->size(1);
    const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
    if (block_per_req > compiled_block_per_req) {
        throw std::runtime_error("piecewise decode: runtime block_tables width exceeds compiled");
    }
    auto &graph_block_tables = graph_input.block_tables.value();
    set_minus_one(graph_block_tables);
    graph_block_tables->narrow({{1, 0, block_per_req}})->copy_from(runtime.block_tables.value());
    graph_input.slot_mapping.value()->copy_from(runtime.slot_mapping.value());
}

std::optional<infinicore::Tensor>
PiecewiseDecodeCompiler::run_decode(const InfinilmModel::Input &input) {
    if (!enabled_ || compiled_.empty()) {
        ++decode_misses_;
        return std::nullopt;
    }
    if (!input.block_tables.has_value() || !input.input_ids.has_value()) {
        ++decode_misses_;
        return std::nullopt;
    }
    const size_t batch = input.block_tables.value()->size(0);
    const size_t input_width = input.input_ids.value()->size(1);
    // Decode only (tokens == requests).
    if (batch != input_width) {
        ++decode_misses_;
        return std::nullopt;
    }
    auto it = compiled_.find(batch);
    if (it == compiled_.end()) {
        ++decode_misses_;
        return std::nullopt;
    }

    auto &batch_graphs = it->second;
    copy_runtime_into_batch_(batch_graphs, input);
    attn_metadata_utils::set_attn_metadata_for_decode_batch(batch_graphs.input, input);

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = batch;
    piecewise.bucket_seq_len = batch;
    piecewise.hidden_states = batch_graphs.hidden_states;
    piecewise.residual = batch_graphs.residual;
    piecewise.ar_staging = batch_graphs.ar_staging;
    piecewise.layer_staging = batch_graphs.layer_staging;
    piecewise.allow_inductor_pre_attn = false;

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(batch_graphs.input, piecewise.hidden_states);

    // MetaX M=1: device-graph replay of 57 tiny segments regresses (~+80ms vs eager
    // piecewise). Default serve path uses eager piecewise; capture still proves
    // has_device_exec for Phase 2. Set INFINI_DECODE_PIECEWISE_REPLAY=graph to force
    // device launches.
    static const bool eager_replay = []() {
        const char *v = std::getenv("INFINI_DECODE_PIECEWISE_REPLAY");
        if (v == nullptr || v[0] == '\0') {
            return true; // latency-first default on MetaX decode M=1
        }
        return std::string(v) == "eager";
    }();

    const size_t num_layers = batch_graphs.pre_attn.size();
    size_t device_runs = 0;
    for (size_t layer = 0; layer < num_layers; ++layer) {
        if (!eager_replay && batch_graphs.pre_attn[layer]) {
            batch_graphs.pre_attn[layer]->run();
            ++segment_replays_;
            if (batch_graphs.pre_attn[layer]->last_replay_used_device()) {
                ++device_runs;
            }
        } else {
            model_->native_piecewise_pre_attn_layer(
                layer, batch_graphs.input, piecewise.hidden_states, piecewise.residual);
        }

        model_->native_piecewise_eager_attn_layer(layer, batch_graphs.input);

        if (!eager_replay && batch_graphs.post_attn[layer]) {
            batch_graphs.post_attn[layer]->run();
            ++segment_replays_;
            if (batch_graphs.post_attn[layer]->last_replay_used_device()) {
                ++device_runs;
            }
        } else {
            model_->native_piecewise_post_attn_decode_cg_layer(
                layer, batch_graphs.input, piecewise.hidden_states, piecewise.residual);
        }

        model_->native_piecewise_eager_moe_layer(
            layer, batch_graphs.input, piecewise.hidden_states, piecewise.residual);
    }

    if (!eager_replay && batch_graphs.lm_head) {
        batch_graphs.lm_head->run();
        ++segment_replays_;
        if (batch_graphs.lm_head->last_replay_used_device()) {
            ++device_runs;
        }
    } else {
        model_->native_piecewise_lm_head(batch_graphs.input,
                                         piecewise.hidden_states,
                                         piecewise.residual,
                                         batch_graphs.logits_holder);
    }

    ++decode_hits_;
    if (decode_hits_ == 1 || decode_hits_ % 32 == 0) {
        spdlog::info(
            "piecewise decode replay: hits={} eager_replay={} segment_replays={} "
            "device_runs_this_step={}",
            decode_hits_,
            eager_replay,
            segment_replays_,
            device_runs);
    }
    return batch_graphs.logits_holder->narrow({{1, 0, batch}});
}

} // namespace infinilm::engine
