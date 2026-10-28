#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cstdint>
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

} // namespace

PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        model_->get_cache_config());
    if (paged_config == nullptr || paged_config->max_batch_size() == 0) {
        return;
    }
    const auto &config = model_->get_model_config();
    const auto type = config ? config->get_or<std::string>("model_type", "") : "";
    lazy_capture_ = infinicore::context::getDevice().getType() == infinicore::Device::Type::ILUVATAR
                 && (type == "qwen3" || type == "qwen3_5" || type == "qwen3_5_text");
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
    const auto *config = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    if (!config || decode_batch_sizes_.empty()) {
        return;
    }
    compiled_map_decode_.clear();
    const size_t max_batch = *std::max_element(decode_batch_sizes_.begin(), decode_batch_sizes_.end());
    block_tables_holder_ = infinicore::Tensor::empty(
        {config->num_blocks() * max_batch}, infinicore::DataType::I32, infinicore::context::getDevice());
    set_zeros(block_tables_holder_);
    if (!lazy_capture_) {
        for (auto b : decode_batch_sizes_) {
            compile_bucket(b, config->num_blocks());
        }
    }
}

void PagedCompiler::compile_bucket(size_t b, size_t block_per_req) {
    const auto *config = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const size_t nblocks = config->num_blocks();
    auto &forward_context = infinilm::global_state::get_forward_context();
    const bool has_mamba_state = has_mamba_cache(forward_context);
    const auto &model_config = model_->get_model_config();
    const size_t position_id_axes = model_config ? model_config->get_or<size_t>("position_id_axes", 1) : 1;
    if (position_id_axes == 0) {
        throw std::runtime_error("PagedCompiler: position_id_axes must be positive");
    }

    // Synthetic capture only touches KV block zero and Mamba slots zero/one.
    // A bucket may be compiled while other requests are live. Restore those
    // slices and metadata even on failure; never advance a live recurrent state.
    struct CacheGuard {
        infinilm::global_state::ForwardContext &ctx;
        infinilm::global_state::AttentionMetadata attn;
        infinilm::global_state::MambaMetadata mamba;
        std::vector<std::pair<infinicore::Tensor, infinicore::Tensor>> saved;
        explicit CacheGuard(infinilm::global_state::ForwardContext &context) : ctx(context), attn(context.attn_metadata), mamba(context.mamba_metadata) {}
        void save(const infinicore::Tensor &view) {
            auto backup = infinicore::Tensor::empty(view->shape(), view->dtype(), view->device());
            backup->copy_from(view);
            saved.emplace_back(view, backup);
        }
        ~CacheGuard() {
            // A failed forward can leave operator recording active. Abandon it
            // before restoring live cache slices, or those copies get recorded
            // instead of executed and the request state remains corrupted.
            if (infinicore::context::isGraphRecording()) {
                infinicore::context::cancelGraphRecording();
            }
            for (auto &[view, backup] : saved) {
                view->copy_from(backup);
            }
            infinicore::context::syncStream();
            ctx.attn_metadata = attn;
            ctx.mamba_metadata = mamba;
        }
    } guard(forward_context);
    if (lazy_capture_) {
        for (const auto &kv : forward_context.kv_cache_vec) {
            if (kv) {
                if (kv->ndim() != 5 || kv->size(0) != 2 || kv->size(1) != nblocks) {
                    throw std::runtime_error("PagedCompiler: unsupported lazy-capture KV layout");
                }
                guard.save(kv->narrow({{1, 0, 1}}));
            }
        }
        for (const auto *states : {&forward_context.conv_state_vec, &forward_context.ssm_state_vec}) {
            for (const auto &state : *states) {
                if (state) {
                    guard.save(state->narrow({{0, 0, 2}}));
                }
            }
        }
    }
    auto make_decode_input = [&](size_t b) {
        InfinilmModel::Input input;
        input.input_ids = infinicore::Tensor::empty({1, b}, infinicore::DataType::I64, infinicore::context::getDevice());
        input.position_ids = infinicore::Tensor::empty(
            position_id_axes > 1
                ? std::vector<size_t>{position_id_axes, b}
                : std::vector<size_t>{b},
            infinicore::DataType::I64, infinicore::context::getDevice());
        input.total_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(input.input_ids.value());
        set_zeros(input.position_ids.value());
        set_zeros(input.total_sequence_lengths.value());
        std::vector<int32_t> total_sequence_lengths_vec(b, 1);
        infinicore::context::memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);
        input.input_offsets = infinicore::Tensor::empty({b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
        std::vector<int32_t> input_offsets_vec(b + 1, 0);
        for (size_t i = 0; i <= b; i++) {
            input_offsets_vec[i] = i;
        }
        infinicore::context::memcpyH2D(input.input_offsets.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
        input.cu_seqlens = infinicore::Tensor::empty({b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
        infinicore::context::memcpyH2D(input.cu_seqlens.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
        input.block_tables = lazy_capture_
                               ? infinicore::Tensor::zeros({b, block_per_req}, infinicore::DataType::I32, infinicore::context::getDevice())
                               : block_tables_holder_->as_strided({b, block_per_req}, {(ptrdiff_t)block_per_req, 1});
        input.slot_mapping = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
        set_zeros(input.slot_mapping.value());

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
        // Hybrid linear-attention layers read cache indices from the same
        // thread-local context. These tensors remain alive in CompiledResult
        // and are updated in place before every graph replay.
        forward_context.mamba_metadata = {
            input.input_offsets,
            input.mamba_init_state_indices,
            input.mamba_final_state_indices,
            std::nullopt,
        };
        return input;
    };

    auto input = make_decode_input(b);

    barrier_->wait();
    (void)model_->forward(input);
    infinicore::context::syncStream();
    // Capture must not start with stale Marlin locks from previous
    // warmup/capture attempts. This reset is intentionally outside
    // graph capture; the current implementation still pays a memset
    // before every graph replay in get_compiled().
    model_->reset_runtime_state();
    infinicore::context::syncStream();
    if (lazy_capture_) {
        // Prefill can leave several GiB of unused blocks in other allocator
        // size classes. Device-graph instantiation/launch allocates driver
        // storage outside that allocator and cannot trigger its OOM retry.
        // Reclaim only unused, unpinned blocks at this cold capture boundary;
        // live tensors and all previously captured graph storage stay valid.
        infinicore::context::trimMemory();
    }
    infinicore::context::startGraphRecording();
    auto output = model_->forward(input);
    auto graph = infinicore::context::stopGraphRecording();
    barrier_->wait();

    auto shared_output = std::shared_ptr<InfinilmModel::Output>(
        new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

    compiled_map_decode_[{b, block_per_req}] = CompiledResult{std::move(input), std::make_tuple(graph, shared_output)};
}

PagedCompiler::Compiled PagedCompiler::get_compiled(const InfinilmModel::Input &input) {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t batch_size = input.block_tables.value()->size(0);
        size_t block_per_req = input.block_tables.value()->size(1);

        // only support decode only batch
        if (batch_size != input.input_ids.value()->size(1)) {
            return {nullptr, nullptr};
        } else {
            const auto *config = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
            size_t capacity = config->num_blocks();
            if (lazy_capture_) {
                // A table spanning the entire cache pool can force a slow
                // attention kernel even when the actual context is short.
                // Keep a 4K-token initial capacity and grow geometrically.
                capacity = std::min(config->num_blocks(), (size_t{4096} + config->block_size() - 1) / config->block_size());
                capacity = std::max(capacity, size_t{1});
                while (capacity < block_per_req && capacity < config->num_blocks()) {
                    capacity = std::min(capacity * 2, config->num_blocks());
                }
                if (block_per_req > capacity) {
                    return {nullptr, nullptr};
                }
            }
            const auto key = std::make_pair(batch_size, capacity);
            auto result = compiled_map_decode_.find(key);
            if (result == compiled_map_decode_.end()) {
                if (!lazy_capture_ || !block_tables_holder_
                    || std::find(decode_batch_sizes_.begin(), decode_batch_sizes_.end(), batch_size) == decode_batch_sizes_.end()) {
                    return {nullptr, nullptr};
                }
                compile_bucket(batch_size, capacity);
                result = compiled_map_decode_.find(key);
            }
            auto &graph_input = result->second.input;

            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
            graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
            graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());

            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
            if (block_per_req > compiled_block_per_req) {
                // Runtime width exceeds compiled graph slot; fall back to eager path.
                return {nullptr, nullptr};
            }

            // Initialize only the active graph rows to -1, then overwrite the
            // runtime logical region. Avoid clearing the full preallocated
            // holder on every decode token.
            auto &graph_block_tables = graph_input.block_tables.value();
            set_minus_one_device_async(graph_block_tables);
            graph_block_tables->narrow({{1, 0, block_per_req}})->copy_from(input.block_tables.value());
            graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());

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

            auto graph = std::get<0>(result->second.compiled);
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
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});

            return std::make_tuple(graph, shared_output);
        }
    } else {
        return {nullptr, nullptr};
    }
}

} // namespace infinilm::engine
