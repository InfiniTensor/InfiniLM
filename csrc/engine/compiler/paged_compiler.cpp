#include "paged_compiler.hpp"
#include "paged_decode_bucket.hpp"
#include "paged_prepare_inputs.hpp"
#include "prepare_state_guard.hpp"
#include "runtime/utils.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace infinilm::engine {
namespace {

constexpr size_t kShortDecodeBlockTableWidth = 8;
constexpr size_t kShortDecodeBlockSize = 256;
constexpr size_t kShortDecodeMaxSequenceLength = kShortDecodeBlockTableWidth * kShortDecodeBlockSize;
constexpr size_t kBoundedDecodePageSize = 256;
bool has_mamba_cache(const infinilm::global_state::ForwardContext &forward_context) {
    auto has_state = [](const std::vector<infinilm::runtime::Tensor> &state_vec) {
        for (const auto &state : state_vec) {
            if (state) {
                return true;
            }
        }
        return false;
    };

    return has_state(forward_context.conv_state_vec) || has_state(forward_context.ssm_state_vec);
}

bool is_cpu_contiguous_tensor(
    const std::optional<infinilm::runtime::Tensor> &tensor,
    infinilm::runtime::DataType dtype,
    const std::vector<size_t> &shape) {
    return tensor.has_value()
        && tensor.value()
        && tensor.value()->device().type() == infinilm::runtime::Device::Type::kCpu
        && tensor.value()->dtype() == dtype
        && tensor.value()->shape() == shape
        && tensor.value()->is_contiguous();
}

bool tensors_compatible(const infinilm::runtime::Tensor &target,
                        const infinilm::runtime::Tensor &source) {
    return target
        && source
        && target->shape() == source->shape()
        && target->dtype() == source->dtype();
}

bool required_tensors_compatible(
    const std::optional<infinilm::runtime::Tensor> &target,
    const std::optional<infinilm::runtime::Tensor> &source) {
    return target.has_value()
        && source.has_value()
        && tensors_compatible(target.value(), source.value());
}

bool optional_tensors_compatible(
    const std::optional<infinilm::runtime::Tensor> &target,
    const std::optional<infinilm::runtime::Tensor> &source) {
    if (target.has_value() != source.has_value()) {
        return false;
    }
    return !target.has_value()
        || tensors_compatible(target.value(), source.value());
}

} // namespace

bool uses_short_decode_graph(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *, infinilm::runtime::Device::Type,
    const PagedGraphConfig &graph_config) {
    graph_config.validate();
    return paged_config.block_size() == kShortDecodeBlockSize
        && paged_config.num_blocks() >= kShortDecodeBlockTableWidth;
}

FinitePagedDecodeProfile make_finite_paged_decode_profile(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,bool ordinary_kv_prepare,
    infinilm::runtime::Device::Type device_type,const PagedGraphConfig &graph_config) {
    graph_config.validate();
    if(model_config==nullptr) return {};
    return {
        ordinary_kv_prepare && graph_config.pp_size == 1,
        graph_config.attention_backend == backends::AttentionBackend::FLASH_ATTN ||
            graph_config.attention_backend == backends::AttentionBackend::PAGED_ATTN,
        model_config->get_kv_quant_scheme() == quantization::KVQuantAlgo::NONE,
        uses_short_decode_graph(paged_config, model_config, device_type, graph_config),
        paged_config.block_size(), paged_config.num_blocks()
    };
}

PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model,
                             RankBarrier *barrier,
                             infinilm::runtime::Runtime &runtime,
                             infinilm::global_state::ForwardContext &forward_context,
                             PagedGraphConfig graph_config)
    : GraphCompiler(model, barrier, runtime, forward_context), graph_config_(graph_config) {
    graph_config_.validate();
    const auto *paged_config = dynamic_cast<const cache::PagedKVCacheConfig *>(
        model_->get_cache_config());
    if (paged_config == nullptr || paged_config->max_batch_size() == 0) {
        return;
    }
    // A decode request needs at least one distinct physical block. Larger
    // batches cannot run on this cache, including during preparation.
    const size_t max_batch_size = std::min(
        paged_config->max_batch_size(), paged_config->num_blocks());
    if (max_batch_size == 0) {
        return;
    }
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
    if (paged_config != nullptr) {
        compiled_short_decode_b1_.reset();
        compiled_map_decode_.clear();
        block_tables_holder_.reset();
        short_block_tables_holder_.reset();

        const size_t nblocks = paged_config->num_blocks();
        auto &forward_context = forward_context_;
        const bool has_mamba_state = has_mamba_cache(forward_context);

        const auto &model_config = model_->get_model_config();
        if (decode_batch_sizes_.empty()) return;
        if (!model_->supports_kv_only_prepare()) {
            skip_unavailable_prepare("paged graph prepare cannot restore this model's state");
            return;
        }
        // Rank-local violations must fail/abort the worker group, not return
        // while peers enter capture barriers. The opt-in above is model-wide.
        if (has_mamba_state) {
            throw std::invalid_argument("KV-only prepare model owns recurrent state");
        }
        const size_t position_id_axes = model_config == nullptr
                                          ? 1
                                          : model_config->get_or<size_t>("position_id_axes", 1);
        if (position_id_axes == 0) {
            throw std::runtime_error("PagedCompiler: position_id_axes must be positive");
        }

        const size_t max_batch_size = *std::max_element(decode_batch_sizes_.begin(), decode_batch_sizes_.end());
        if (nblocks > static_cast<size_t>(std::numeric_limits<int32_t>::max())
            || nblocks > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) / max_batch_size) {
            throw std::invalid_argument("paged prepare block table size overflows");
        }
        auto write_views = paged_prepare_views(
            runtime_, forward_context, nblocks, paged_config->block_size(), max_batch_size, 1);
        PrepareStateGuard prepare_state{runtime_, forward_context, std::move(write_views)};
        decltype(compiled_map_decode_) candidate_decode;
        decltype(compiled_short_decode_b1_) candidate_short_decode;
        infinilm::runtime::Tensor candidate_short_holder;
        auto candidate_holder = infinilm::runtime::Tensor::empty(
            runtime_, {nblocks * max_batch_size}, infinilm::runtime::DataType::kInt32);

        auto make_decode_input = [&](size_t b,
                                     size_t block_per_req,
                                     const infinilm::runtime::Tensor &block_tables_holder,
                                     size_t logical_bound) {
            InfinilmModel::Input input;
            input.logical_kv_upper_bound = logical_bound;
            input.input_ids = infinilm::runtime::Tensor::empty(runtime_, {1, b}, infinilm::runtime::DataType::kInt64);
            input.position_ids = infinilm::runtime::Tensor::empty(
                runtime_,
                position_id_axes > 1
                    ? std::vector<size_t>{position_id_axes, b}
                    : std::vector<size_t>{b},
                infinilm::runtime::DataType::kInt64);
            input.total_sequence_lengths = infinilm::runtime::Tensor::empty(runtime_, {b}, infinilm::runtime::DataType::kInt32);
            runtime_.setDeviceMemory(input.input_ids.value()->data(), 0, input.input_ids.value()->nbytes());
            runtime_.setDeviceMemory(input.position_ids.value()->data(), 0, input.position_ids.value()->nbytes());
            runtime_.setDeviceMemory(input.total_sequence_lengths.value()->data(), 0, input.total_sequence_lengths.value()->nbytes());
            std::vector<int32_t> total_sequence_lengths_vec(b, 1);
            runtime_.memcpyH2D(input.total_sequence_lengths.value()->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t), false);
            input.input_offsets = infinilm::runtime::Tensor::empty(runtime_, {b + 1}, infinilm::runtime::DataType::kInt32);
            std::vector<int32_t> input_offsets_vec(b + 1, 0);
            for (size_t i = 0; i <= b; i++) {
                input_offsets_vec[i] = i;
            }
            runtime_.memcpyH2D(input.input_offsets.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            input.cu_seqlens = infinilm::runtime::Tensor::empty(runtime_, {b + 1}, infinilm::runtime::DataType::kInt32);
            runtime_.memcpyH2D(input.cu_seqlens.value()->data(), input_offsets_vec.data(), (b + 1) * sizeof(int32_t), false);
            input.block_tables = block_tables_holder->as_strided(
                {b, block_per_req},
                {static_cast<ptrdiff_t>(block_per_req), 1});
            input.slot_mapping = infinilm::runtime::Tensor::empty(runtime_, {b}, infinilm::runtime::DataType::kInt64);
            initialize_paged_prepare_metadata(
                runtime_, input.block_tables.value(), input.slot_mapping.value(),
                nblocks, paged_config->block_size());

            // Only ordinary KV state is qualified by this prepare contract.
            forward_context.attn_metadata = {
                input.past_sequence_lengths,
                input.total_sequence_lengths,
                input.input_offsets,
                input.cu_seqlens,
                input.block_tables,
                input.slot_mapping,
            };
            forward_context.attn_metadata.logical_kv_upper_bound = logical_bound;
            forward_context.mamba_metadata = {
                input.input_offsets,
                std::nullopt,
                std::nullopt,
            };
            return input;
        };

        auto make_compiled_result = [this](
                                        InfinilmModel::Input input,
                                        std::shared_ptr<infinilm::execution::graph::Graph> graph,
                                        const InfinilmModel::Output &output) {
            auto shared_output = snapshot_output(output);

            return CompiledResult{
                std::move(input),
                std::make_tuple(std::move(graph), std::move(shared_output))};
        };

        auto capture_decode = [&](size_t b,
                                  size_t block_per_req,
                                  const infinilm::runtime::Tensor &block_tables_holder,
                                  size_t logical_bound) {
            auto input = make_decode_input(b, block_per_req,
                                           block_tables_holder, logical_bound);

            try {
            barrier_->wait();
            (void)model_->forward(input, forward_context_);
            runtime_.syncStream();
            GraphRecordingGuard recording{runtime_};
            auto output = model_->forward(input, forward_context_);
            auto graph = recording.finish();
            barrier_->wait();

            return make_compiled_result(
                std::move(input), std::move(graph), output);
            } catch (...) {
                try { runtime_.syncStream(); } catch (...) { std::terminate(); }
                throw;
            }
        };

        for (size_t b : decode_batch_sizes_) {
            candidate_decode.emplace(
                std::make_pair(b, size_t{0}),
                capture_decode(b, nblocks, candidate_holder, 0));
        }

        const auto finite_profile=make_finite_paged_decode_profile(
            *paged_config,model_config.get(),model_->supports_kv_only_prepare(),
            runtime_.device().type(),graph_config_);
        if (supports_finite_paged_decode_bucket(finite_profile)) {
            for (size_t b : {size_t{1}, size_t{2}}) {
                if (b > max_batch_size) continue;
                for (size_t bound : {size_t{256}, size_t{512}}) {
                    candidate_decode.emplace(
                        std::make_pair(b, bound),
                        capture_decode(b, bound / kBoundedDecodePageSize,
                                       candidate_holder, bound));
                }
            }

        }

        if (uses_short_decode_graph(
                *paged_config, model_->get_model_config().get(),
                runtime_.device().type(), graph_config_)) {
            candidate_short_holder = infinilm::runtime::Tensor::empty(
                runtime_,
                {kShortDecodeBlockTableWidth},
                infinilm::runtime::DataType::kInt32);
            candidate_short_decode.emplace(capture_decode(
                1,
                kShortDecodeBlockTableWidth,
                candidate_short_holder,
                0));
        }

        // No graph becomes available until every touched slot and metadata
        // reference has been restored. Failure destroys candidates, not a
        // partially published collection of plans.
        prepare_state.restore();
        compiled_map_decode_.swap(candidate_decode);
        compiled_short_decode_b1_.swap(candidate_short_decode);
        block_tables_holder_ = std::move(candidate_holder);
        short_block_tables_holder_ = std::move(candidate_short_holder);
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

    CompiledResult *selected_result = nullptr;
    bool use_short_decode_graph = false;
    size_t required_pages = 0;
    {
        // Compiled paged plans are decode-only.
        if (!is_cpu_contiguous_tensor(input.input_offsets,
                                      infinilm::runtime::DataType::kInt32,
                                      {batch_size + 1})
            || !is_packed_paged_decode(
                input.input_ids.value()->size(0),
                input.input_ids.value()->size(1),
                reinterpret_cast<const int32_t *>(input.input_offsets.value()->data()),
                batch_size + 1, batch_size)) {
            return {nullptr, nullptr};
        }

        const size_t logical_bound = input.logical_kv_upper_bound;
        if (logical_bound != 0) {
            if (batch_size > 2
                || (logical_bound != 256 && logical_bound != 512)
                || block_per_req != logical_bound / kBoundedDecodePageSize
                || !input.total_sequence_lengths.has_value()
                || !input.total_sequence_lengths.value()
                || input.total_sequence_lengths.value()->device().type()
                       != infinilm::runtime::Device::Type::kCpu
                || input.total_sequence_lengths.value()->dtype()
                       != infinilm::runtime::DataType::kInt32
                || !input.total_sequence_lengths.value()->is_contiguous()
                || input.total_sequence_lengths.value()->ndim() != 1
                || input.total_sequence_lengths.value()->size(0) != batch_size) {
                return {nullptr, nullptr};
            }
            const auto *lengths = reinterpret_cast<const int32_t *>(
                input.total_sequence_lengths.value()->data());
            for (size_t row = 0; row < batch_size; ++row) {
                if (lengths[row] <= 0
                    || static_cast<size_t>(lengths[row]) > logical_bound) {
                    return {nullptr, nullptr};
                }
            }
        }
        auto general_result = compiled_map_decode_.find(
            std::make_pair(batch_size, logical_bound));
        if (general_result == compiled_map_decode_.end()) {
            return {nullptr, nullptr};
        }
        selected_result = &general_result->second;

        if (logical_bound == 0 && batch_size == 1
            && compiled_short_decode_b1_.has_value()) {
            const auto &total_sequence_lengths = input.total_sequence_lengths;
            const bool valid_short_metadata = total_sequence_lengths.has_value()
                                           && total_sequence_lengths.value()
                                           && total_sequence_lengths.value()->device().type()
                                                  == infinilm::runtime::Device::Type::kCpu
                                           && total_sequence_lengths.value()->dtype()
                                                  == infinilm::runtime::DataType::kInt32
                                           && total_sequence_lengths.value()->ndim() == 1
                                           && total_sequence_lengths.value()->size(0) == 1
                                           && total_sequence_lengths.value()->is_contiguous()
                                           && runtime_block_tables->device().type()
                                                  == infinilm::runtime::Device::Type::kCpu
                                           && runtime_block_tables->dtype() == infinilm::runtime::DataType::kInt32
                                           && runtime_block_tables->is_contiguous();
            if (valid_short_metadata) {
                const int32_t total_sequence_length = reinterpret_cast<const int32_t *>(
                    total_sequence_lengths.value()->data())[0];
                if (total_sequence_length > 0
                    && static_cast<size_t>(total_sequence_length)
                           <= kShortDecodeMaxSequenceLength) {
                    required_pages = 1
                                   + (static_cast<size_t>(total_sequence_length) - 1)
                                         / paged_config->block_size();
                    if (required_pages <= kShortDecodeBlockTableWidth
                        && block_per_req >= required_pages) {
                        selected_result = &compiled_short_decode_b1_.value();
                        use_short_decode_graph = true;
                    }
                }
        }
    }
}

    auto &graph_input = selected_result->input;
            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
    if ((!use_short_decode_graph && block_per_req > compiled_block_per_req)
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

    graph_input.input_ids.value()->copy_from(runtime_, input.input_ids.value());
    graph_input.position_ids.value()->copy_from(runtime_, input.position_ids.value());
    graph_input.total_sequence_lengths.value()->copy_from(runtime_, input.total_sequence_lengths.value());
    graph_input.input_offsets.value()->copy_from(runtime_, input.input_offsets.value());
    graph_input.cu_seqlens.value()->copy_from(runtime_, input.cu_seqlens.value());

    auto &graph_block_tables = graph_input.block_tables.value();
    if (use_short_decode_graph) {
        runtime_.setDeviceMemoryAsync(graph_block_tables->data(), 0xFF,
                                      graph_block_tables->nbytes(), runtime_.stream());
        runtime_.memcpyH2D(
            graph_block_tables->data(),
            runtime_block_tables->data(),
            required_pages * sizeof(int32_t),
            false);
    } else {
        // Initialize only the active graph rows to -1, then overwrite the
        // runtime logical region. Avoid clearing the full preallocated holder
        // on every decode token.
        runtime_.setDeviceMemoryAsync(graph_block_tables->data(), 0xFF,
                                      graph_block_tables->nbytes(), runtime_.stream());
        graph_block_tables->narrow({{1, 0, block_per_req}})
            ->copy_from(runtime_, runtime_block_tables);
    }
    graph_input.slot_mapping.value()->copy_from(runtime_, input.slot_mapping.value());

    auto graph = std::get<0>(selected_result->compiled);
    const auto &compiled_output = std::get<1>(selected_result->compiled);
    auto shared_output = resume_output(*compiled_output);

            return std::make_tuple(graph, shared_output);
        }

} // namespace infinilm::engine
