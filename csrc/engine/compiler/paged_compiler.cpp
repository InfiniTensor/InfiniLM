#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
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

bool tensors_compatible(const infinicore::Tensor &target,
                        const infinicore::Tensor &source) {
    return target
        && source
        && target->shape() == source->shape()
        && target->dtype() == source->dtype();
}

bool required_tensors_compatible(
    const std::optional<infinicore::Tensor> &target,
    const std::optional<infinicore::Tensor> &source) {
    return target.has_value()
        && source.has_value()
        && tensors_compatible(target.value(), source.value());
}

bool optional_tensors_compatible(
    const std::optional<infinicore::Tensor> &target,
    const std::optional<infinicore::Tensor> &source) {
    if (target.has_value() != source.has_value()) {
        return false;
    }
    return !target.has_value()
        || tensors_compatible(target.value(), source.value());
}

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
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        model_->get_cache_config());
    if (paged_config != nullptr && !decode_batch_sizes_.empty()) {
        compiled_map_decode_.clear();
        block_tables_holder_.reset();

        const size_t nblocks = paged_config->num_blocks();
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
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks * max_batch_size}, infinicore::DataType::kInt32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);

        auto make_decode_input = [&](size_t b,
                                     size_t block_per_req,
                                     const infinicore::Tensor &block_tables_holder) {
            InfinilmModel::Input input;
            input.input_ids = infinicore::Tensor::empty({1, b}, infinicore::DataType::kInt64, infinicore::context::getDevice());
            input.position_ids = infinicore::Tensor::empty(
                position_id_axes > 1
                    ? std::vector<size_t>{position_id_axes, b}
                    : std::vector<size_t>{b},
                infinicore::DataType::kInt64, infinicore::context::getDevice());
            input.total_sequence_lengths = infinicore::Tensor::empty({b}, infinicore::DataType::kInt32, infinicore::context::getDevice());
            set_zeros(input.input_ids.value());
            set_zeros(input.position_ids.value());
            set_zeros(input.total_sequence_lengths.value());
            std::vector<int32_t> total_sequence_lengths_vec(b, 1);
            infinicore::context::memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);
            input.input_offsets = infinicore::Tensor::empty({b + 1}, infinicore::DataType::kInt32, infinicore::context::getDevice());
            std::vector<int32_t> input_offsets_vec(b + 1, 0);
            for (size_t i = 0; i <= b; i++) {
                input_offsets_vec[i] = i;
            }
            infinicore::context::memcpyH2D(input.input_offsets.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            input.cu_seqlens = infinicore::Tensor::empty({b + 1}, infinicore::DataType::kInt32, infinicore::context::getDevice());
            infinicore::context::memcpyH2D(input.cu_seqlens.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            input.block_tables = block_tables_holder->as_strided(
                {b, block_per_req},
                {static_cast<ptrdiff_t>(block_per_req), 1});
            input.slot_mapping = infinicore::Tensor::empty({b}, infinicore::DataType::kInt64, infinicore::context::getDevice());
            set_zeros(input.slot_mapping.value());

            if (has_mamba_state) {
                input.mamba_init_state_indices = infinicore::Tensor::empty(
                    {b}, infinicore::DataType::kInt32, infinicore::context::getDevice());
                input.mamba_final_state_indices = infinicore::Tensor::empty(
                    {b}, infinicore::DataType::kInt32, infinicore::context::getDevice());
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
            };
            return input;
        };

        auto make_compiled_result = [](
                                        InfinilmModel::Input input,
                                        std::shared_ptr<infinicore::graph::Graph> graph,
                                        const InfinilmModel::Output &output) {
            infinicore::Tensor graph_hidden_states;
            if (output.hidden_states) {
                graph_hidden_states = infinicore::graph::GraphTensor(
                    output.hidden_states,
                    infinicore::graph::GraphTensor::SnapshotPolicy::kBlob);
            }
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{
                    infinicore::graph::GraphTensor(
                        output.logits,
                        infinicore::graph::GraphTensor::SnapshotPolicy::kBlob),
                    graph_hidden_states});

            return CompiledResult{
                std::move(input),
                std::make_tuple(std::move(graph), std::move(shared_output))};
        };

        auto capture_decode = [&](size_t b,
                                  size_t block_per_req,
                                  const infinicore::Tensor &block_tables_holder) {
            auto input = make_decode_input(b, block_per_req, block_tables_holder);

            barrier_->wait();
            (void)model_->forward(input);
            infinicore::context::syncStream();
            // Capture must not start with stale state from previous attempts.
            model_->reset_runtime_state();
            infinicore::context::syncStream();
            GraphRecordingGuard recording;
            auto output = model_->forward(input);
            auto graph = recording.finish();
            barrier_->wait();

            return make_compiled_result(
                std::move(input), std::move(graph), output);
        };

        {
            const size_t warmup_batch_size = std::min(max_batch_size, static_cast<size_t>(64));
            auto input = make_decode_input(
                warmup_batch_size, nblocks, block_tables_holder_);
            model_->forward(input);
            infinicore::context::syncStream();
            // Clear transient operator state before CUDA graph capture.
            model_->reset_runtime_state();
            infinicore::context::syncStream();
        }

        for (size_t b : decode_batch_sizes_) {
            compiled_map_decode_[b] = capture_decode(b, nblocks, block_tables_holder_);
        }
    }
}

PagedCompiler::Compiled PagedCompiler::get_compiled(const InfinilmModel::Input &input) {
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        model_->get_cache_config());
    if (paged_config == nullptr
        || !input.block_tables.has_value()
        || !input.block_tables.value()
        || input.block_tables.value()->ndim() != 2
        || !input.input_ids.has_value()
        || !input.input_ids.value()
        || input.input_ids.value()->ndim() != 2) {
        return {nullptr, nullptr};
    }

    const auto &runtime_block_tables = input.block_tables.value();
    const size_t batch_size = runtime_block_tables->size(0);
    const size_t block_per_req = runtime_block_tables->size(1);

    // Only single-token decode batches have a captured graph.
    if (batch_size != input.input_ids.value()->size(1)
        || input.sample_all_positions || input.target_hidden_states.has_value()
        || input.pixel_values.has_value()) {
        return {nullptr, nullptr};
    }
    auto result = compiled_map_decode_.find(batch_size);
    if (result == compiled_map_decode_.end()) {
        return {nullptr, nullptr};
    }
    auto *selected_result = &result->second;

    auto &graph_input = selected_result->input;
    const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
    if (block_per_req > compiled_block_per_req
        || graph_input.block_tables.value()->dtype()
               != runtime_block_tables->dtype()
        || graph_input.block_tables.value()->size(0) != batch_size
        || !required_tensors_compatible(graph_input.input_ids, input.input_ids)
        || !required_tensors_compatible(
            graph_input.position_ids, input.position_ids)
        || !required_tensors_compatible(
            graph_input.total_sequence_lengths,
            input.total_sequence_lengths)
        || !required_tensors_compatible(
            graph_input.input_offsets, input.input_offsets)
        || !required_tensors_compatible(
            graph_input.cu_seqlens, input.cu_seqlens)
        || !required_tensors_compatible(
            graph_input.slot_mapping, input.slot_mapping)
        || !optional_tensors_compatible(
            graph_input.mamba_init_state_indices,
            input.mamba_init_state_indices)
        || !optional_tensors_compatible(
            graph_input.mamba_final_state_indices,
            input.mamba_final_state_indices)) {
        // Validate every input before mutating storage shared by a graph.
        return {nullptr, nullptr};
    }

    graph_input.input_ids.value()->copy_from(input.input_ids.value());
    graph_input.position_ids.value()->copy_from(input.position_ids.value());
    graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
    graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
    graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());

    auto &graph_block_tables = graph_input.block_tables.value();
    set_minus_one_device_async(graph_block_tables);
    graph_block_tables->narrow({{1, 0, block_per_req}})
        ->copy_from(runtime_block_tables);
    graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());

    const bool graph_has_mamba_indices = graph_input.mamba_init_state_indices.has_value()
                                      && graph_input.mamba_final_state_indices.has_value();
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

    auto graph = std::get<0>(selected_result->compiled);
    const auto &compiled_output = std::get<1>(selected_result->compiled);
    infinicore::Tensor hidden_states;
    if (compiled_output->hidden_states) {
        hidden_states = compiled_output->hidden_states->resume_from_blob_();
    }
    auto shared_output = std::shared_ptr<InfinilmModel::Output>(
        new InfinilmModel::Output{
            compiled_output->logits->resume_from_blob_(),
            hidden_states});

    return std::make_tuple(graph, shared_output);
}

} // namespace infinilm::engine
