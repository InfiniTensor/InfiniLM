#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../layers/attention/backends/attention_layer.hpp"
#include "../../utils.hpp"

#include <infinicore/ops/random_sample.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <vector>

namespace infinilm::engine {
namespace {

bool has_mamba_cache(const infinilm::global_state::ForwardContext &forward_context) {
    auto has_state = [](const std::vector<infinicore::Tensor> &state_vec) {
        for (const auto &state : state_vec) {
            if (state) {
                return true;
            }
        }
        return false;
    };

    return has_state(forward_context.conv_state_vec) || has_state(forward_context.ssm_state_vec);
}

// Wraps the per-request greedy-sampling loop as one recordable operator.
// infinicore::op::random_sample_ has no graph-recording hook of its own (it
// runs eagerly even while recording), so without this wrapper a companion
// sampling graph would record an empty op list and replay as a no-op.
class SamplingLoopOperator : public infinicore::graph::GraphOperator {
public:
    explicit SamplingLoopOperator(std::function<void()> fn) : fn_(std::move(fn)) {}
    void run() const override { fn_(); }

private:
    std::function<void()> fn_;
};

} // namespace

PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        model_->get_cache_config());
    if (paged_config == nullptr || paged_config->max_batch_size() == 0) {
        return;
    }
    const size_t max_batch_size = paged_config->max_batch_size();
    auto append_batch_size = [&](size_t batch_size) {
        if (batch_size <= max_batch_size) {
            decode_batch_sizes_.push_back(batch_size);
        }
    };

    for (size_t b = 1; b < 64; ++b) {
        append_batch_size(b);
    }
    for (size_t b = 64; b < 128; b += 16) {
        append_batch_size(b);
    }
    for (size_t b = 128; b < 256; b += 32) {
        append_batch_size(b);
    }
    for (size_t b = 256; b <= 512; b += 64) {
        append_batch_size(b);
    }
    if (decode_batch_sizes_.empty() || decode_batch_sizes_.back() != max_batch_size) {
        decode_batch_sizes_.push_back(max_batch_size);
    }
}

void PagedCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
        auto &forward_context = infinilm::global_state::get_forward_context();
        const bool has_mamba_state = has_mamba_cache(forward_context);

        const auto &model_config = model_->get_model_config();
        const size_t position_id_axes = model_config == nullptr
                                          ? 1
                                          : model_config->get_or<size_t>("position_id_axes", 1);
        if (position_id_axes == 0) {
            throw std::runtime_error("PagedCompiler: position_id_axes must be positive");
        }

        size_t max_batch_size = *std::max_element(decode_batch_sizes_.begin(), decode_batch_sizes_.end());
        compiled_map_decode_.clear();
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks * max_batch_size}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);

        auto make_decode_input = [&](size_t b, size_t fake_max_seq_len, CompiledResult *cr) {
            InfinilmModel::Input input;
            // The small per-step inputs live as views into two contiguous
            // device buffers, so get_compiled() can refresh them with two
            // packed H2D copies instead of one copy per tensor.
            //   pack_i64: input_ids (b) | position_ids (axes*b) | slot_mapping (b)
            //   pack_i32: total_seq_lens (b) | input_offsets (b+1) | cu_seqlens (b+1)
            auto pack_i64 = infinicore::Tensor::empty({(position_id_axes + 2) * b}, infinicore::DataType::I64, infinicore::context::getDevice());
            auto pack_i32 = infinicore::Tensor::empty({3 * b + 2}, infinicore::DataType::I32, infinicore::context::getDevice());
            set_zeros(pack_i64);
            {
                std::vector<int32_t> init_i32(3 * b + 2, 0);
                for (size_t i = 0; i < b; i++) {
                    init_i32[i] = 1;                 // total_sequence_lengths
                    init_i32[b + i] = i;             // input_offsets
                    init_i32[2 * b + 1 + i] = i;     // cu_seqlens
                }
                init_i32[2 * b] = b;                 // input_offsets[b]
                init_i32[3 * b + 1] = b;             // cu_seqlens[b]
                infinicore::context::memcpyH2D(pack_i32->data(), init_i32.data(), init_i32.size() * sizeof(int32_t), false);
            }
            input.input_ids = pack_i64->narrow({{0, 0, b}})->view({1, b});
            input.position_ids = position_id_axes > 1
                                     ? pack_i64->narrow({{0, b, position_id_axes * b}})->view({position_id_axes, b})
                                     : pack_i64->narrow({{0, b, b}});
            input.slot_mapping = pack_i64->narrow({{0, (position_id_axes + 1) * b, b}});
            input.total_sequence_lengths = pack_i32->narrow({{0, 0, b}});
            input.input_offsets = pack_i32->narrow({{0, b, b + 1}});
            input.cu_seqlens = pack_i32->narrow({{0, 2 * b + 1, b + 1}});
            const size_t block_per_req = nblocks;
            input.block_tables = block_tables_holder_->as_strided({b, block_per_req}, {(ptrdiff_t)block_per_req, 1});

            if (has_mamba_state) {
                input.mamba_init_state_indices = infinicore::Tensor::empty(
                    {b}, infinicore::DataType::I32, infinicore::context::getDevice());
                input.mamba_final_state_indices = infinicore::Tensor::empty(
                    {b}, infinicore::DataType::I32, infinicore::context::getDevice());
                std::vector<int32_t> init_state_indices_vec(b, 0);
                std::vector<int32_t> final_state_indices_vec(b, 1);
                infinicore::context::memcpyH2D(
                    input.mamba_init_state_indices.value()->data(),
                    init_state_indices_vec.data(),
                    b * sizeof(int32_t),
                    false);
                infinicore::context::memcpyH2D(
                    input.mamba_final_state_indices.value()->data(),
                    final_state_indices_vec.data(),
                    b * sizeof(int32_t),
                    false);
            }

            // Attention reads attn_metadata from thread-local forward context.
            forward_context.attn_metadata = {
                input.past_sequence_lengths,
                input.total_sequence_lengths,
                input.input_offsets,
                input.cu_seqlens,
                input.block_tables,
                input.slot_mapping,
            };
            // Decode-kernel routing hint for the hybrid attention layer. The
            // recorded op list freezes whichever kernel this selects, so the
            // long-ctx capture passes a value above the routing threshold
            // (1 = always the splitkv variant).
            forward_context.attn_metadata.max_sequence_length = fake_max_seq_len;
            // Hybrid linear-attention layers read cache indices from the same
            // thread-local context. These tensors remain alive in CompiledResult
            // and are updated in place before every graph replay.
            forward_context.mamba_metadata = {
                input.input_offsets,
                input.mamba_init_state_indices,
                input.mamba_final_state_indices,
            };
            if (cr != nullptr) {
                cr->pack_i64 = pack_i64;
                cr->pack_i32 = pack_i32;
                cr->stage_i64.assign((position_id_axes + 2) * b, 0);
                cr->stage_i32.assign(3 * b + 2, 0);
                cr->position_id_axes = position_id_axes;
            }
            return input;
        };

        {
            const size_t warmup_batch_size = std::min(max_batch_size, static_cast<size_t>(64));
            auto input = make_decode_input(warmup_batch_size, 1, nullptr);
            model_->forward(input);
            infinicore::context::syncStream();
            // Warmup runs the eager Marlin path and may leave per-layer lock
            // workspaces dirty. Reset before CUDA graph capture so capture
            // starts from the same all-zero lock state as normal execution.
            model_->reset_runtime_state();
            infinicore::context::syncStream();
        }

        // Only HYBRID routes decode between two kernels by ctx length; for
        // every other backend the long-ctx variant would be a bit-identical
        // copy of the short one, so skip the second capture and its memory.
        if (infinilm::global_state::get_infinilm_config().attention_backend == ::infinilm::backends::AttentionBackend::HYBRID) {
            decode_ctx_threshold_ = layers::attention::backends::decode_fa_ctx_threshold();
        }

        for (size_t b : decode_batch_sizes_) {
            DecodeVariants variants;

            // fake_max_seq_len only steers the hybrid layer's decode-kernel
            // routing while recording; the captured kernels read the real
            // per-request lengths from pack_i32 at replay.
            auto capture_variant = [&](size_t fake_max_seq_len, CompiledResult &cr) {
                // Both barrier waits must stay unconditional: with TP>1 the
                // ranks are threads sharing this host-side barrier, so a rank
                // that fails mid-capture has to arrive at both waits anyway
                // before the error propagates, or the other ranks block
                // forever at mismatched barrier generations.
                barrier_->wait();
                bool recording = false;
                try {
                    auto input = make_decode_input(b, fake_max_seq_len, &cr);

                    (void)model_->forward(input);
                    infinicore::context::syncStream();
                    // Capture must not start with stale Marlin locks from previous
                    // warmup/capture attempts. This reset is intentionally outside
                    // graph capture; the current implementation still pays a memset
                    // before every graph replay in get_compiled().
                    model_->reset_runtime_state();
                    infinicore::context::syncStream();
                    infinicore::context::startGraphRecording();
                    recording = true;
                    auto output = model_->forward(input);
                    auto graph = infinicore::context::stopGraphRecording();
                    recording = false;

                    auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                        new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

                    cr.input = std::move(input);
                    cr.compiled = std::make_tuple(graph, shared_output);

                    // Capture the greedy-sampling kernels (one cub ArgMax + index
                    // cast per request, reading the decode graph's logits blob)
                    // into a companion graph. Greedy batches replay this instead
                    // of issuing ~3 kernel launches per request per step.
                    // Best-effort: any capture failure just falls back to the
                    // eager sampling loop. Limited to small batch sizes to bound
                    // capture time at load.
                    if (b <= 64) {
                        try {
                            const size_t vocab_size = output.logits->shape().back();
                            auto logits2d = output.logits->view({b, vocab_size});
                            auto sampling_out = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
                            // top_k == 1 / temperature == 0 both take the argmax branch
                            // in the sampling op, so the captured kernels match greedy
                            // semantics and never consume random_val. Captured by
                            // value: the operator outlives this scope and keeps both
                            // tensors alive.
                            auto run_sampling = [logits2d, sampling_out, b, vocab_size]() {
                                for (size_t i = 0; i < b; ++i) {
                                    auto score = logits2d->narrow({{0, i, 1}})->view({vocab_size});
                                    auto out = sampling_out->narrow({{0, i, 1}})->view({});
                                    infinicore::op::random_sample_(out, score, 0.0f, 1.0f, 1, 0.0f);
                                }
                            };
                            // Recording alone does not run the op; instantiate() warms
                            // the loop (settling descriptor/workspace allocations)
                            // before capturing it into a device graph segment.
                            infinicore::context::startGraphRecording();
                            infinicore::context::addGraphOperator(std::make_shared<SamplingLoopOperator>(run_sampling));
                            cr.sampling_graph = infinicore::context::stopGraphRecording();
                            cr.sampling_out = sampling_out;
                        } catch (const std::exception &e) {
                            spdlog::warn("PagedCompiler: sampling graph capture failed for batch {}: {}", b, e.what());
                            cr.sampling_graph = nullptr;
                            cr.sampling_out = {};
                        }
                    }
                } catch (...) {
                    if (recording) {
                        // Discard the dangling partial recording so later
                        // captures start from a clean state.
                        try {
                            (void)infinicore::context::stopGraphRecording();
                        } catch (...) {
                        }
                    }
                    barrier_->wait();
                    throw;
                }
                barrier_->wait();
            };

            capture_variant(1, variants.short_ctx);
            if (decode_ctx_threshold_ != std::numeric_limits<size_t>::max()) {
                // Second capture with the FA kvcache decode kernel recorded.
                // Best-effort: the short-ctx variant alone is still correct.
                try {
                    capture_variant(decode_ctx_threshold_ + 1, variants.long_ctx);
                    variants.has_long = true;
                } catch (const std::exception &e) {
                    spdlog::warn("PagedCompiler: long-ctx decode graph capture failed for batch {}: {}", b, e.what());
                    variants.has_long = false;
                }
            }
            compiled_map_decode_[b] = std::move(variants);
        }
    }
}

PagedCompiler::Compiled PagedCompiler::get_compiled(const InfinilmModel::Input &input) {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t batch_size = input.block_tables.value()->size(0);
        size_t block_per_req = input.block_tables.value()->size(1);

        // only support decode only batch
        if (batch_size != input.input_ids.value()->size(1)) {
            return {nullptr, nullptr};
        } else {
            auto result = compiled_map_decode_.find(batch_size);
            if (result == compiled_map_decode_.end()) {
                return {nullptr, nullptr};
            }
            auto &variants = result->second;

            const auto &rt_input_ids = input.input_ids.value();
            const auto &rt_position_ids = input.position_ids.value();
            const auto &rt_total_lens = input.total_sequence_lengths.value();
            const auto &rt_offsets = input.input_offsets.value();
            const auto &rt_cu_seqlens = input.cu_seqlens.value();
            const auto &rt_slots = input.slot_mapping.value();

            // Pick the decode-kernel variant by the batch's longest context.
            // The lengths tensor is expected on the host (graph replay
            // requires a CPU int32 tensor below anyway); anything else
            // conservatively routes to the short-ctx (splitkv) variant.
            size_t max_ctx = 0;
            if (variants.has_long
                && rt_total_lens->device().getType() == infinicore::Device::Type::CPU
                && rt_total_lens->dtype() == infinicore::DataType::I32
                && rt_total_lens->is_contiguous()
                && rt_total_lens->shape().size() == 1
                && rt_total_lens->shape()[0] == batch_size) {
                const auto *lens = reinterpret_cast<const int32_t *>(rt_total_lens->data());
                for (size_t i = 0; i < batch_size; ++i) {
                    max_ctx = std::max(max_ctx, static_cast<size_t>(lens[i]));
                }
            }
            auto &cr = (variants.has_long && max_ctx > decode_ctx_threshold_)
                         ? variants.long_ctx
                         : variants.short_ctx;
            variants.last_served = &cr;
            auto &graph_input = cr.input;

            // Fast path: pack the six small inputs host-side and refresh the
            // graph inputs with two H2D copies. Host pointers only: anything
            // living on a device must go through copy_from below.
            const bool packable =
                rt_input_ids->device().getType() == infinicore::Device::Type::CPU
                && rt_position_ids->device().getType() == infinicore::Device::Type::CPU
                && rt_total_lens->device().getType() == infinicore::Device::Type::CPU
                && rt_offsets->device().getType() == infinicore::Device::Type::CPU
                && rt_cu_seqlens->device().getType() == infinicore::Device::Type::CPU
                && rt_slots->device().getType() == infinicore::Device::Type::CPU
                && rt_input_ids->is_contiguous() && rt_position_ids->is_contiguous()
                && rt_total_lens->is_contiguous() && rt_offsets->is_contiguous()
                && rt_cu_seqlens->is_contiguous() && rt_slots->is_contiguous()
                && rt_input_ids->dtype() == infinicore::DataType::I64
                && rt_position_ids->dtype() == infinicore::DataType::I64
                && rt_slots->dtype() == infinicore::DataType::I64
                && rt_total_lens->dtype() == infinicore::DataType::I32
                && rt_offsets->dtype() == infinicore::DataType::I32
                && rt_cu_seqlens->dtype() == infinicore::DataType::I32
                && rt_input_ids->numel() == batch_size
                && rt_position_ids->numel() == cr.position_id_axes * batch_size
                && rt_total_lens->numel() == batch_size
                && rt_offsets->numel() == batch_size + 1
                && rt_cu_seqlens->numel() == batch_size + 1
                && rt_slots->numel() == batch_size;
            if (packable) {
                std::memcpy(cr.stage_i64.data(), rt_input_ids->data(), batch_size * sizeof(int64_t));
                std::memcpy(cr.stage_i64.data() + batch_size, rt_position_ids->data(), cr.position_id_axes * batch_size * sizeof(int64_t));
                std::memcpy(cr.stage_i64.data() + (cr.position_id_axes + 1) * batch_size, rt_slots->data(), batch_size * sizeof(int64_t));
                infinicore::context::memcpyH2D(cr.pack_i64->data(), cr.stage_i64.data(), cr.stage_i64.size() * sizeof(int64_t));

                std::memcpy(cr.stage_i32.data(), rt_total_lens->data(), batch_size * sizeof(int32_t));
                std::memcpy(cr.stage_i32.data() + batch_size, rt_offsets->data(), (batch_size + 1) * sizeof(int32_t));
                std::memcpy(cr.stage_i32.data() + 2 * batch_size + 1, rt_cu_seqlens->data(), (batch_size + 1) * sizeof(int32_t));
                infinicore::context::memcpyH2D(cr.pack_i32->data(), cr.stage_i32.data(), cr.stage_i32.size() * sizeof(int32_t));
            } else {
                graph_input.input_ids.value()->copy_from(rt_input_ids);
                graph_input.position_ids.value()->copy_from(rt_position_ids);
                graph_input.total_sequence_lengths.value()->copy_from(rt_total_lens);
                graph_input.input_offsets.value()->copy_from(rt_offsets);
                graph_input.cu_seqlens.value()->copy_from(rt_cu_seqlens);
                graph_input.slot_mapping.value()->copy_from(rt_slots);
            }

            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
            if (block_per_req > compiled_block_per_req) {
                // Runtime width exceeds compiled graph slot; fall back to eager path.
                return {nullptr, nullptr};
            }

            // Initialize only the active graph rows to -1, then overwrite the
            // runtime logical region. Avoid clearing the full preallocated
            // holder on every decode token. When the runtime block table
            // already covers the compiled width, the copy below overwrites
            // every column and the fill is redundant.
            auto &graph_block_tables = graph_input.block_tables.value();
            if (block_per_req < compiled_block_per_req) {
                set_minus_one_device_async(graph_block_tables);
            }
            graph_block_tables->narrow({{1, 0, block_per_req}})->copy_from(input.block_tables.value());

            const bool graph_has_mamba_indices = graph_input.mamba_init_state_indices.has_value() && graph_input.mamba_final_state_indices.has_value();
            const bool input_has_mamba_indices = input.mamba_init_state_indices.has_value() && input.mamba_final_state_indices.has_value();
            if (graph_has_mamba_indices != input_has_mamba_indices) {
                return {nullptr, nullptr};
            }
            if (graph_has_mamba_indices) {
                graph_input.mamba_init_state_indices.value()->copy_from(
                    input.mamba_init_state_indices.value());
                graph_input.mamba_final_state_indices.value()->copy_from(
                    input.mamba_final_state_indices.value());
            }
            // CUDA graph replay reuses the same per-layer Marlin workspaces.
            // The graph itself does not contain a workspace reset, so enqueue
            // one on the same stream before launch. This is correct but costs
            // decode latency; the intended follow-up is a reusable global
            // zero workspace/lock buffer shared by all Marlin layers.
            model_->reset_runtime_state();

            auto graph = std::get<0>(cr.compiled);
            if (graph != nullptr) {
                const auto &runtime_seq_lens = input.total_sequence_lengths.value();
                if (runtime_seq_lens->device().getType()
                        != infinicore::Device::Type::CPU
                    || runtime_seq_lens->dtype() != infinicore::DataType::I32
                    || runtime_seq_lens->shape().size() != 1
                    || runtime_seq_lens->shape()[0] != batch_size) {
                    throw std::runtime_error(
                        "PagedCompiler expected CPU int32 "
                        "total_sequence_lengths for graph replay");
                }
                graph->bind_host_int_array(
                    graph_input.total_sequence_lengths.value(),
                    reinterpret_cast<const int32_t *>(
                        runtime_seq_lens->data()),
                    batch_size);
            }
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(cr.compiled)->logits->resume_from_blob_()});

            return std::make_tuple(graph, shared_output);
        }
    } else {
        return {nullptr, nullptr};
    }
}

std::pair<std::shared_ptr<infinicore::graph::Graph>, infinicore::Tensor>
PagedCompiler::get_sampling_compiled(size_t batch_size) {
    // Kill switch for A/B measurement: force the eager per-request sampling loop.
    static const bool disabled = std::getenv("INFINILM_DISABLE_SAMPLING_GRAPH") != nullptr;
    if (disabled) {
        return {nullptr, {}};
    }
    auto it = compiled_map_decode_.find(batch_size);
    if (it == compiled_map_decode_.end()) {
        return {nullptr, {}};
    }
    // Serve the sampling graph of whichever decode variant get_compiled()
    // most recently handed out for this batch size (they read different
    // logits blobs).
    const CompiledResult *cr = it->second.last_served != nullptr
                                 ? it->second.last_served
                                 : &it->second.short_ctx;
    if (cr->sampling_graph == nullptr) {
        return {nullptr, {}};
    }
    return {cr->sampling_graph, cr->sampling_out};
}

} // namespace infinilm::engine
