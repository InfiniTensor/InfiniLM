#include "prepare_state_guard.hpp"
#include "static_batching_compiler.hpp"
#include "../../cache/cache.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
size_t static_graph_configured_page_size(
    const infinilm::cache::StaticKVCacheConfig &cache_config,
    const infinilm::config::ModelConfig &model_config) {
    const size_t configured_page_size = cache_config.max_cache_len();
    return configured_page_size == 0
               || configured_page_size == std::numeric_limits<size_t>::max()
           ? model_config.get<size_t>("max_position_embeddings")
           : configured_page_size;
}

bool supports_static_graph_configuration(
    const infinilm::cache::StaticKVCacheConfig &cache_config,
    const infinilm::config::ModelConfig &model_config,
    const infinilm::runtime::Device &device) {
    if (device.type() != infinilm::runtime::Device::Type::kNvidia
        || model_config.get_kv_quant_scheme() != infinilm::quantization::KVQuantAlgo::NONE) {
        return false;
    }
    const auto dtype = model_config.get_kv_cache_dtype();
    const auto head_dim = model_config.get_head_dim();
    const size_t page_size = static_graph_configured_page_size(
        cache_config, model_config);
    return cache_config.max_batch_size() > 0
        && (dtype == infinilm::runtime::DataType::kFloat16
            || dtype == infinilm::runtime::DataType::kBFloat16)
        && (head_dim == 64 || head_dim == 128)
        && page_size > 0 && page_size % 256 == 0;
}

size_t require_static_graph_local_cache(
    infinilm::runtime::Runtime &runtime,
    size_t cache_capacity,
    const infinilm::global_state::ForwardContext &forward_context,
    infinilm::runtime::DataType expected_dtype,
    size_t expected_head_dim,
    size_t expected_page_size) {
    // Model/config eligibility is uniform across ranks. Once eligible, a
    // malformed local cache or recurrent state must fail the worker so its
    // RankBarrier is aborted; silently skipping can strand another rank.
    if (forward_context.runtime != &runtime) {
        throw std::invalid_argument("static graph requires its local ForwardContext Runtime");
    }
    const auto views = infinilm::engine::static_prepare_views(
        runtime, forward_context);
    (void)views;
    const auto &kv_cache_vec = forward_context.kv_cache_vec;
    for (const auto &kv_cache : kv_cache_vec) {
        if (kv_cache->size(1) != cache_capacity
            || kv_cache->size(3) != expected_page_size
            || kv_cache->dtype() != expected_dtype
            || kv_cache->size(4) != expected_head_dim) {
            throw std::invalid_argument("static graph requires matching local KV layouts");
        }
    }
    return expected_page_size;
}
} // namespace

namespace infinilm::engine {
StaticBatchingCompiler::StaticBatchingCompiler(const std::shared_ptr<InfinilmModel> &model,
                                               RankBarrier *barrier,
                                               infinilm::runtime::Runtime &runtime,
                                               infinilm::global_state::ForwardContext &forward_context)
    : GraphCompiler(model, barrier, runtime, forward_context) {
}

void StaticBatchingCompiler::compile() {
    compiled_map_.clear();
    const auto *static_config = dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config());
    if (static_config == nullptr) {
        return;
    }

    const size_t cache_capacity = static_config->max_batch_size();
    const auto &model_config = model_->get_model_config();
    if (!model_->supports_kv_only_prepare()
        || !model_config
        || !supports_static_graph_configuration(
            *static_config, *model_config, runtime_.device())) {
        return;
    }
    const size_t cache_page_size = require_static_graph_local_cache(
        runtime_, cache_capacity, forward_context_, model_config->get_kv_cache_dtype(),
        model_config->get_head_dim(),
        static_graph_configured_page_size(*static_config, *model_config));
    const bool uses_target_hidden_states = model_config
                                        && model_config->get_or<std::string>("model_type", "") == "minicpm_eagle";
    decltype(compiled_map_) candidate;
    std::vector<size_t> batches{1};
    if (cache_capacity != 1) batches.push_back(cache_capacity);
    for (size_t b : batches) {
        PrepareStateGuard prepare_state{
            runtime_, forward_context_, static_prepare_views(runtime_, forward_context_)};
        InfinilmModel::Input input;
        input.input_ids = infinilm::runtime::Tensor::empty(runtime_, {b, 1}, infinilm::runtime::DataType::kInt64);
        input.position_ids = infinilm::runtime::Tensor::empty(runtime_, {b, 1}, infinilm::runtime::DataType::kInt64);
        input.past_sequence_lengths = infinilm::runtime::Tensor::empty(runtime_, {b}, infinilm::runtime::DataType::kInt32);
        input.total_sequence_lengths = infinilm::runtime::Tensor::empty(runtime_, {b}, infinilm::runtime::DataType::kInt32);
        runtime_.setDeviceMemory(input.input_ids.value()->data(), 0, input.input_ids.value()->nbytes());
        runtime_.setDeviceMemory(input.position_ids.value()->data(), 0, input.position_ids.value()->nbytes());
        runtime_.setDeviceMemory(input.past_sequence_lengths.value()->data(), 0, input.past_sequence_lengths.value()->nbytes());
        std::vector<int32_t> total_sequence_lengths_vec(b, 1);
        runtime_.memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);
        StaticDecodeMetadata metadata;
        metadata.prepare(runtime_, b, cache_capacity, cache_page_size);
        input.block_tables = metadata.block_tables();
        input.slot_mapping = metadata.slot_mapping();
        if (uses_target_hidden_states) {
            input.target_hidden_states = infinilm::runtime::Tensor::empty(
                runtime_, {b, 1, model_config->get<size_t>("hidden_size")},
                model_config->get_dtype());
            runtime_.setDeviceMemory(input.target_hidden_states.value()->data(), 0,
                                     input.target_hidden_states.value()->nbytes());
        }

        forward_context_.attn_metadata = {
            input.past_sequence_lengths,
            input.total_sequence_lengths,
            input.input_offsets,
            input.cu_seqlens,
            input.block_tables,
            input.slot_mapping,
        };
        forward_context_.attn_metadata.first_past_sequence_length = 0;
        forward_context_.attn_metadata.first_total_sequence_length = 1;
        forward_context_.attn_metadata.static_decode_host =
            global_state::StaticDecodeHostDescription{true, true};

        barrier_->wait();
        (void)model_->forward(input, forward_context_);
        runtime_.syncStream();

        GraphRecordingGuard recording{runtime_};
        auto output = model_->forward(input, forward_context_);
        auto graph = recording.finish();
        barrier_->wait();

        infinilm::runtime::Tensor graph_hidden_states;
        if (output.hidden_states) {
            graph_hidden_states = infinilm::execution::graph::GraphTensor(
                runtime_,
                output.hidden_states,
                infinilm::execution::graph::GraphTensor::SnapshotPolicy::kBlob);
        }
        auto shared_output = std::shared_ptr<InfinilmModel::Output>(
            new InfinilmModel::Output{
                infinilm::execution::graph::GraphTensor(
                    runtime_,
                    output.logits,
                    infinilm::execution::graph::GraphTensor::SnapshotPolicy::kBlob),
                graph_hidden_states});

        prepare_state.restore();
        candidate[std::make_tuple(b, 1)] = CompiledResult{
            std::move(input), std::make_tuple(graph, shared_output), std::move(metadata)};
    }
    compiled_map_.swap(candidate);
}

StaticBatchingCompiler::Compiled StaticBatchingCompiler::get_compiled(
    const InfinilmModel::Input &input) {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::StaticKVCacheConfig *>(model_->get_cache_config())) {
        if (!input.input_ids || input.input_ids.value()->ndim() != 2) {
            throw std::invalid_argument("static decode graph requires input_ids[batch,1]");
        }
        size_t batch_size = input.input_ids.value()->size(0);
        size_t seqlen = input.input_ids.value()->size(1);
        auto result = compiled_map_.find(std::make_tuple(batch_size, seqlen));
        if (result == compiled_map_.end()) {
            return std::make_tuple(nullptr, nullptr);
        } else {
            auto &graph_input = result->second.input;
            const auto copy_source_device_allowed = [this](const infinilm::runtime::Tensor &source) {
                return source && (source->device() == runtime_.device()
                    || source->device().type() == infinilm::runtime::Device::Type::kCpu);
            };
            const bool graph_has_target_hidden_states = graph_input.target_hidden_states.has_value();
            const bool input_has_target_hidden_states = input.target_hidden_states.has_value();
            if (graph_has_target_hidden_states != input_has_target_hidden_states) {
                return std::make_tuple(nullptr, nullptr);
            }
            if (graph_has_target_hidden_states
                && (!graph_input.target_hidden_states.value()
                    || !copy_source_device_allowed(input.target_hidden_states.value())
                    || graph_input.target_hidden_states.value()->shape()
                           != input.target_hidden_states.value()->shape()
                    || graph_input.target_hidden_states.value()->dtype()
                           != input.target_hidden_states.value()->dtype())) {
                return std::make_tuple(nullptr, nullptr);
            }
            const auto same_layout = [&copy_source_device_allowed](
                                         const infinilm::runtime::Tensor &target,
                                         const std::optional<infinilm::runtime::Tensor> &source) {
                return source && copy_source_device_allowed(*source)
                    && target->shape() == (*source)->shape()
                    && target->dtype() == (*source)->dtype();
            };
            if (!same_layout(*graph_input.input_ids, input.input_ids)
                || !same_layout(*graph_input.position_ids, input.position_ids)
                || !same_layout(*graph_input.past_sequence_lengths, input.past_sequence_lengths)
                || !same_layout(*graph_input.total_sequence_lengths, input.total_sequence_lengths)) {
                throw std::invalid_argument("static decode graph input layout mismatch");
            }
            if (input.input_offsets) {
                const auto &offsets = *input.input_offsets;
                if (!offsets || offsets->device().type() != infinilm::runtime::Device::Type::kCpu
                    || offsets->dtype() != infinilm::runtime::DataType::kInt32
                    || offsets->ndim() != 1 || offsets->size(0) != batch_size + 1
                    || !offsets->is_contiguous()) {
                    throw std::invalid_argument("static decode input_offsets require CPU int32[batch+1]");
                }
                const auto *values = reinterpret_cast<const int32_t *>(offsets->data());
                for (size_t i = 0; i <= batch_size; ++i) {
                    if (values[i] != static_cast<int32_t>(i)) {
                        throw std::invalid_argument("static decode requires one input token per request");
                    }
                }
            }
            // Validate all host metadata before copying any captured graph input.
            // This also updates stable block/slot buffers for the selected plan.
            result->second.metadata.update(
                runtime_, *input.past_sequence_lengths, *input.total_sequence_lengths,
                input.block_tables, input.slot_mapping);
            graph_input.input_ids.value()->copy_from(runtime_, input.input_ids.value());
            graph_input.position_ids.value()->copy_from(runtime_, input.position_ids.value());
            graph_input.past_sequence_lengths.value()->copy_from(runtime_, input.past_sequence_lengths.value());
            graph_input.total_sequence_lengths.value()->copy_from(runtime_, input.total_sequence_lengths.value());
            if (graph_has_target_hidden_states) {
                graph_input.target_hidden_states.value()->copy_from(
                    runtime_, input.target_hidden_states.value());
            }

            auto graph = std::get<0>(result->second.compiled);
            const auto &compiled_output = std::get<1>(result->second.compiled);
            infinilm::runtime::Tensor hidden_states;
            if (compiled_output->hidden_states) {
                hidden_states = compiled_output->hidden_states->resume_from_blob_(runtime_);
            }
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{
                    compiled_output->logits->resume_from_blob_(runtime_),
                    hidden_states});
            return std::make_tuple(graph, shared_output);
        }
    } else {
        return std::make_tuple(nullptr, nullptr);
    }
}
} // namespace infinilm::engine
