#include "piecewise_decode_compiler.hpp"

#include "../../global_state/decode_phase_profile.hpp"
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
        batches = {1};
    }
    std::sort(batches.begin(), batches.end());
    batches.erase(std::unique(batches.begin(), batches.end()), batches.end());
    return batches;
}

/// 0 (default) = span fuse: FA outside recording; MoE host-break inside
///                 record(post_i + moe_i + pre_{i+1}). Graph::run ~30; device
///                 segments still ~57 while MoE is host-break.
///   1 = legacy per-layer pre/post (FA+MoE fully outside recording).
///   2 = post-only CG: eager pre+FA+MoE; device-graph only post (+lm_head).
///       Cuts device launches ~57→29 (MetaX M=1 launch-tax experiment).
size_t parse_fuse_layers_() {
    const char *raw = std::getenv("INFINI_DECODE_PIECEWISE_FUSE_LAYERS");
    if (raw == nullptr || raw[0] == '\0') {
        return 0;
    }
    return static_cast<size_t>(std::stoul(raw));
}

size_t count_device_segments_(const std::shared_ptr<infinicore::graph::Graph> &g) {
    if (!g) {
        return 0;
    }
    return g->device_segment_count();
}

struct RecGuard {
    bool active{true};
    RecGuard() { infinicore::context::startGraphRecording(); }
    std::shared_ptr<infinicore::graph::Graph> stop() {
        active = false;
        return infinicore::context::stopGraphRecording();
    }
    ~RecGuard() {
        if (active) {
            try {
                (void)infinicore::context::stopGraphRecording();
            } catch (...) {
            }
        }
    }
};

} // namespace

PiecewiseDecodeCompiler::PiecewiseDecodeCompiler(const std::shared_ptr<InfinilmModel> &model,
                                                 RankBarrier *barrier)
    : model_(model), barrier_(barrier) {
    enabled_ = native_piecewise_decode_enabled() && model_->supports_native_piecewise_prefill();
    if (!enabled_) {
        return;
    }
    capture_batches_ = parse_capture_batches_();
    fuse_layers_ = parse_fuse_layers_();
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

void PiecewiseDecodeCompiler::capture_batch_legacy_(size_t /*batch*/,
                                                    BatchGraphs &graphs,
                                                    size_t capture_layers) {
    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;
    const bool need_barrier = barrier_ != nullptr && barrier_->num_ranks() > 1;

    graphs.pre_attn.resize(capture_layers);
    graphs.post_attn.resize(capture_layers);

    size_t device_segs = 0;
    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;

        if (need_barrier) {
            barrier_->wait("piecewise_decode_capture_pre_attn");
        }
        {
            RecGuard rec;
            model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
            graphs.pre_attn[layer] = rec.stop();
        }
        device_segs += count_device_segments_(graphs.pre_attn[layer]);

        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        if (need_barrier) {
            barrier_->wait("piecewise_decode_capture_post_attn");
        }
        {
            RecGuard rec;
            model_->native_piecewise_post_attn_decode_cg_layer(
                layer, graphs.input, hidden, residual);
            graphs.post_attn[layer] = rec.stop();
        }
        device_segs += count_device_segments_(graphs.post_attn[layer]);

        model_->native_piecewise_eager_moe_layer(layer, graphs.input, hidden, residual);
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
    if (need_barrier) {
        barrier_->wait("piecewise_decode_capture_lm_head");
    }
    {
        RecGuard rec;
        model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
        graphs.lm_head = rec.stop();
    }
    device_segs += count_device_segments_(graphs.lm_head);
    graphs.device_segments = device_segs;
}

void PiecewiseDecodeCompiler::capture_batch_fused_(size_t /*batch*/,
                                                   BatchGraphs &graphs,
                                                   size_t capture_layers) {
    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;
    const bool need_barrier = barrier_ != nullptr && barrier_->num_ranks() > 1;
    size_t device_segs = 0;

    // Mode 2: post-only device graphs — skip pre CG to cut launches ~57→29.
    if (fuse_layers_ == 2) {
        graphs.post_attn.resize(capture_layers);
        for (size_t layer = 0; layer < capture_layers; ++layer) {
            piecewise.active_layer = layer;
            piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;
            model_->native_piecewise_pre_attn_layer(layer, graphs.input, hidden, residual);
            piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
            model_->native_piecewise_eager_attn_layer(layer, graphs.input);

            piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
            if (need_barrier) {
                barrier_->wait("piecewise_decode_capture_post_only");
            }
            {
                RecGuard rec;
                model_->native_piecewise_post_attn_decode_cg_layer(
                    layer, graphs.input, hidden, residual);
                graphs.post_attn[layer] = rec.stop();
            }
            device_segs += count_device_segments_(graphs.post_attn[layer]);
            model_->native_piecewise_eager_moe_layer(layer, graphs.input, hidden, residual);
        }
        piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
        {
            RecGuard rec;
            model_->native_piecewise_lm_head(
                graphs.input, hidden, residual, graphs.logits_holder);
            graphs.lm_head = rec.stop();
        }
        device_segs += count_device_segments_(graphs.lm_head);
        graphs.device_segments = device_segs;
        graphs.fuse_layers = 2;
        return;
    }

    // Mode 0: span fusion (MetaX-safe): FA never under isGraphRecording.
    // record(pre_0); then for each layer: FA; record(post + MoE_hostbreak [+ pre_next]).
    piecewise.active_layer = 0;
    piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;
    if (need_barrier) {
        barrier_->wait("piecewise_decode_capture_pre0");
    }
    {
        RecGuard rec;
        model_->native_piecewise_pre_attn_layer(0, graphs.input, hidden, residual);
        auto g = rec.stop();
        device_segs += count_device_segments_(g);
        graphs.layer_groups.push_back(std::move(g));
        graphs.group_layer0.push_back(0);
    }

    for (size_t layer = 0; layer < capture_layers; ++layer) {
        piecewise.active_layer = layer;
        piecewise.phase = global_state::PiecewiseCapturePhase::EagerAttn;
        model_->native_piecewise_eager_attn_layer(layer, graphs.input);

        piecewise.phase = global_state::PiecewiseCapturePhase::PostAttn;
        if (need_barrier) {
            barrier_->wait("piecewise_decode_capture_span");
        }
        {
            RecGuard rec;
            model_->native_piecewise_post_attn_decode_cg_layer(
                layer, graphs.input, hidden, residual);
            model_->native_piecewise_eager_moe_layer(layer, graphs.input, hidden, residual);
            if (layer + 1 < capture_layers) {
                piecewise.active_layer = layer + 1;
                piecewise.phase = global_state::PiecewiseCapturePhase::PreAttn;
                model_->native_piecewise_pre_attn_layer(
                    layer + 1, graphs.input, hidden, residual);
            }
            auto g = rec.stop();
            device_segs += count_device_segments_(g);
            graphs.layer_groups.push_back(std::move(g));
            graphs.group_layer0.push_back(layer);
        }
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::LmHead;
    if (need_barrier) {
        barrier_->wait("piecewise_decode_capture_lm_head");
    }
    {
        RecGuard rec;
        model_->native_piecewise_lm_head(graphs.input, hidden, residual, graphs.logits_holder);
        graphs.lm_head = rec.stop();
    }
    device_segs += count_device_segments_(graphs.lm_head);
    graphs.device_segments = device_segs;
}

void PiecewiseDecodeCompiler::capture_batch_(size_t batch) {
    const auto rank_device = infinilm::global_state::get_tensor_model_parallel_rank_info().device;
    infinicore::context::setDevice(rank_device);
    infinicore::context::InferencePhaseGuard phase_guard(
        infinicore::context::InferencePhase::Decode);

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
    graphs.fuse_layers = fuse_layers_;

    auto &piecewise = infinilm::global_state::get_forward_context().piecewise;
    piecewise.valid_seq_len = batch;
    piecewise.allow_inductor_pre_attn = false;
    piecewise.phase = global_state::PiecewiseCapturePhase::None;

    auto &hidden = piecewise.hidden_states;
    auto &residual = piecewise.residual;

    size_t capture_layers = num_layers;
    if (const char *raw = std::getenv("INFINI_NATIVE_CG_MAX_LAYERS")) {
        capture_layers = std::min(num_layers, static_cast<size_t>(std::stoul(raw)));
    }
    graphs.capture_layers = capture_layers;

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

    set_zeros(piecewise.residual);
    model_->native_piecewise_embed(graphs.input, hidden);

    const bool use_legacy = (fuse_layers_ == 1);
    if (use_legacy) {
        capture_batch_legacy_(batch, graphs, capture_layers);
    } else {
        capture_batch_fused_(batch, graphs, capture_layers);
    }

    piecewise.phase = global_state::PiecewiseCapturePhase::None;
    graphs.hidden_states = piecewise.hidden_states;
    graphs.residual = piecewise.residual;
    graphs.ar_staging = piecewise.ar_staging;
    graphs.layer_staging = piecewise.layer_staging;
    device_segments_captured_ += graphs.device_segments;
    compiled_[batch] = std::move(graphs);

    const auto &stored = compiled_[batch];
    const bool triton_capture = []() {
        const char *v = std::getenv("INFINI_MOE_TRITON_CAPTURE");
        return v != nullptr && v[0] != '\0' && std::string(v) != "0";
    }();
    const bool capture_safe = []() {
        const char *v = std::getenv("INFINI_MOE_CAPTURE_SAFE");
        return v != nullptr && v[0] != '\0' && std::string(v) != "0";
    }();
    const char *mode_name = "span_fuse_moe_hostbreak";
    if (triton_capture) {
        mode_name = "span_fuse_triton_capture";
    } else if (capture_safe) {
        mode_name = "span_fuse_capture_safe";
    }
    if (stored.fuse_layers == 1) {
        mode_name = "legacy_split";
    } else if (stored.fuse_layers == 2) {
        if (triton_capture) {
            mode_name = "post_only_triton_capture";
        } else if (capture_safe) {
            mode_name = "post_only_capture_safe";
        } else {
            mode_name = "post_only_cg";
        }
    }
    spdlog::info(
        "native piecewise decode CG: captured batch={} layers={} fuse_layers={} "
        "layer_groups={} device_segments={} mode={}",
        batch,
        capture_layers,
        stored.fuse_layers,
        stored.layer_groups.size(),
        stored.device_segments,
        mode_name);
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

    spdlog::info("piecewise decode CG: compiling batches=[{}] fuse_layers={}", [&]() {
        std::ostringstream oss;
        for (size_t i = 0; i < capture_batches_.size(); ++i) {
            if (i) {
                oss << ',';
            }
            oss << capture_batches_[i];
        }
        return oss.str();
    }(),
                 fuse_layers_);

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
    // Full wipe only when runtime width is narrower (padding slots must be -1).
    // Decode MC=1 usually matches compiled width — skip the extra kernel.
    if (block_per_req < compiled_block_per_req) {
        set_minus_one(graph_block_tables);
    }
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

    static const bool eager_replay = []() {
        const char *v = std::getenv("INFINI_DECODE_PIECEWISE_REPLAY");
        if (v == nullptr || v[0] == '\0') {
            return true;
        }
        return std::string(v) == "eager";
    }();

    size_t device_runs = 0;
    const bool span_fused = batch_graphs.fuse_layers == 0 && !batch_graphs.layer_groups.empty();
    const bool post_only = batch_graphs.fuse_layers == 2;

    if (span_fused) {
        if (eager_replay) {
            const size_t num_layers = batch_graphs.capture_layers;
            for (size_t layer = 0; layer < num_layers; ++layer) {
                model_->native_piecewise_pre_attn_layer(
                    layer, batch_graphs.input, piecewise.hidden_states, piecewise.residual);
                model_->native_piecewise_eager_attn_layer(layer, batch_graphs.input);
                model_->native_piecewise_post_attn_decode_cg_layer(
                    layer, batch_graphs.input, piecewise.hidden_states, piecewise.residual);
                model_->native_piecewise_eager_moe_layer(
                    layer, batch_graphs.input, piecewise.hidden_states, piecewise.residual);
            }
            model_->native_piecewise_lm_head(batch_graphs.input,
                                             piecewise.hidden_states,
                                             piecewise.residual,
                                             batch_graphs.logits_holder);
        } else {
            // Graph replay: FA host-break between device segments. Profile splits
            // attn_ms (eager FA) vs graph_run_ms (Graph::run = post+MoE[+pre]).
            const bool profile = global_state::decode_phase_profile::recording();
            const bool exclusive =
                profile && global_state::decode_phase_profile::exclusive_sync();
            auto profile_sync = [exclusive]() {
                if (exclusive) {
                    infinicore::context::syncStream();
                }
            };
            auto add_graph_ms = [profile](double t0) {
                if (profile) {
                    global_state::decode_phase_profile::counters().graph_run_ms +=
                        global_state::decode_phase_profile::monotonic_ms() - t0;
                }
            };
            auto add_attn_ms = [profile](double t0) {
                if (profile) {
                    global_state::decode_phase_profile::counters().attn_ms +=
                        global_state::decode_phase_profile::monotonic_ms() - t0;
                }
            };

            if (!batch_graphs.layer_groups.empty() && batch_graphs.layer_groups[0]) {
                const double t0 =
                    profile ? global_state::decode_phase_profile::monotonic_ms() : 0.0;
                batch_graphs.layer_groups[0]->run();
                profile_sync();
                add_graph_ms(t0);
                if (profile) {
                    ++global_state::decode_phase_profile::counters().n_graph_runs;
                }
                ++segment_replays_;
                if (batch_graphs.layer_groups[0]->last_replay_used_device()) {
                    device_runs += batch_graphs.layer_groups[0]->device_segment_count();
                }
            }
            const size_t num_layers = batch_graphs.capture_layers;
            for (size_t layer = 0; layer < num_layers; ++layer) {
                const double t_attn =
                    profile ? global_state::decode_phase_profile::monotonic_ms() : 0.0;
                model_->native_piecewise_eager_attn_layer(layer, batch_graphs.input);
                profile_sync();
                add_attn_ms(t_attn);
                if (profile) {
                    ++global_state::decode_phase_profile::counters().n_fa;
                }
                const size_t gi = layer + 1;
                if (gi < batch_graphs.layer_groups.size() && batch_graphs.layer_groups[gi]) {
                    const double t_g =
                        profile ? global_state::decode_phase_profile::monotonic_ms() : 0.0;
                    batch_graphs.layer_groups[gi]->run();
                    profile_sync();
                    add_graph_ms(t_g);
                    if (profile) {
                        ++global_state::decode_phase_profile::counters().n_graph_runs;
                    }
                    ++segment_replays_;
                    if (batch_graphs.layer_groups[gi]->last_replay_used_device()) {
                        device_runs += batch_graphs.layer_groups[gi]->device_segment_count();
                    }
                }
            }
            if (batch_graphs.lm_head) {
                const double t_lm =
                    profile ? global_state::decode_phase_profile::monotonic_ms() : 0.0;
                batch_graphs.lm_head->run();
                profile_sync();
                add_graph_ms(t_lm);
                if (profile) {
                    ++global_state::decode_phase_profile::counters().n_graph_runs;
                }
                ++segment_replays_;
                if (batch_graphs.lm_head->last_replay_used_device()) {
                    device_runs += batch_graphs.lm_head->device_segment_count();
                }
            }
        }
    } else if (post_only) {
        const size_t num_layers = batch_graphs.capture_layers;
        for (size_t layer = 0; layer < num_layers; ++layer) {
            model_->native_piecewise_pre_attn_layer(
                layer, batch_graphs.input, piecewise.hidden_states, piecewise.residual);
            model_->native_piecewise_eager_attn_layer(layer, batch_graphs.input);
            if (!eager_replay && layer < batch_graphs.post_attn.size()
                && batch_graphs.post_attn[layer]) {
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
    } else {
        const size_t num_layers = batch_graphs.pre_attn.size();
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
    }

    ++decode_hits_;
    if (decode_hits_ == 1 || decode_hits_ % 32 == 0) {
        spdlog::info(
            "piecewise decode replay: hits={} eager_replay={} fuse_layers={} segment_replays={} "
            "device_runs_this_step={} device_segments_captured={}",
            decode_hits_,
            eager_replay,
            batch_graphs.fuse_layers,
            segment_replays_,
            device_runs,
            batch_graphs.device_segments);
    }
    return batch_graphs.logits_holder->narrow({{1, 0, batch}});
}

} // namespace infinilm::engine
