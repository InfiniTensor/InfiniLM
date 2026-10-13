#include "paged_compiler.hpp"
#include "paged_decode_bucket.hpp"
#include "paged_prepare_inputs.hpp"
#include "prepare_state_guard.hpp"
#include <infini/rt/device_guard.h>
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
constexpr size_t kBoundedDecodeMaxPlans = 4;
constexpr size_t kBoundedDecodePinnedBudgetBytes = 512ull * 1024 * 1024;
constexpr size_t kBoundedDecodeFreeWaterlineBytes = 512ull * 1024 * 1024;

constexpr char kBaichuanFixedPrefillGraphEnv[] = "INFINILM_ENABLE_BAICHUAN_PREFILL_GRAPH";
constexpr size_t kBaichuanFixedPrefillBatchSize = 1;
constexpr size_t kBaichuanFixedPrefillSequenceLength = 10;
constexpr size_t kBaichuanFixedPrefillBlockSize = 256;

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

bool env_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::string_view(value) == "1";
}

bool supports_baichuan_fixed_prefill_graph(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,
    bool has_mamba_state,
    infinilm::runtime::Device::Type device_type,
    const PagedGraphConfig &graph_config) {
    if (!env_flag_enabled(kBaichuanFixedPrefillGraphEnv) || has_mamba_state) {
        return false;
    }

    graph_config.validate();
    if (model_config == nullptr) {
        return false;
    }

    const size_t hidden_size = model_config->get_or<size_t>("hidden_size", 0);
    const size_t num_attention_heads = model_config->get_or<size_t>("num_attention_heads", 0);
    const size_t head_dim = model_config->get_or<size_t>(
        "head_dim",
        num_attention_heads == 0 ? 0 : hidden_size / num_attention_heads);
    (void)model_config->get_dtype();
    const auto quant_scheme = model_config->get_quant_scheme();
    const auto kv_quant_scheme = model_config->get_kv_quant_scheme();

    return device_type == infinilm::runtime::Device::Type::kNvidia
        && graph_config.attention_backend == backends::AttentionBackend::FLASH_ATTN
        && model_config->get_or<std::string>("model_type", "") == "baichuan"
        && hidden_size == 4096
        && num_attention_heads == 32
        && model_config->get_or<size_t>("num_key_value_heads", 0) == 32
        && head_dim == 128
        && paged_config.block_size() == kBaichuanFixedPrefillBlockSize
        && paged_config.num_blocks() == 1
        && paged_config.max_batch_size() == 1
        && graph_config.tp_size == 2
        && model_config->get_or<size_t>("num_hidden_layers", 0) == 32
        && model_config->get_or<size_t>("position_id_axes", 1) == 1
        && quant_scheme == quantization::QuantScheme::NONE
        && kv_quant_scheme == quantization::KVQuantAlgo::NONE;
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

template <typename T>
bool tensor_values_equal(
    const std::optional<infinilm::runtime::Tensor> &tensor,
    std::initializer_list<T> expected) {
    const auto *values = reinterpret_cast<const T *>(tensor.value()->data());
    return std::equal(expected.begin(), expected.end(), values);
}

bool is_exact_baichuan_fixed_prefill_input(
    const InfinilmModel::Input &input) {
    const bool tensors_match = is_cpu_contiguous_tensor(
                                   input.input_ids,
                                   infinilm::runtime::DataType::kInt64,
                                   {kBaichuanFixedPrefillBatchSize,
                                    kBaichuanFixedPrefillSequenceLength})
                            && is_cpu_contiguous_tensor(
                                   input.position_ids,
                                   infinilm::runtime::DataType::kInt64,
                                   {kBaichuanFixedPrefillSequenceLength})
                            && is_cpu_contiguous_tensor(
                                   input.past_sequence_lengths,
                                   infinilm::runtime::DataType::kInt32,
                                   {kBaichuanFixedPrefillBatchSize})
                            && is_cpu_contiguous_tensor(
                                   input.total_sequence_lengths,
                                   infinilm::runtime::DataType::kInt32,
                                   {kBaichuanFixedPrefillBatchSize})
                            && is_cpu_contiguous_tensor(
                                   input.input_offsets,
                                   infinilm::runtime::DataType::kInt32,
                                   {kBaichuanFixedPrefillBatchSize + 1})
                            && is_cpu_contiguous_tensor(
                                   input.cu_seqlens,
                                   infinilm::runtime::DataType::kInt32,
                                   {kBaichuanFixedPrefillBatchSize + 1})
                            && is_cpu_contiguous_tensor(
                                   input.block_tables,
                                   infinilm::runtime::DataType::kInt32,
                                   {kBaichuanFixedPrefillBatchSize, 1})
                            && is_cpu_contiguous_tensor(
                                   input.slot_mapping,
                                   infinilm::runtime::DataType::kInt64,
                                   {kBaichuanFixedPrefillSequenceLength});
    if (!tensors_match) {
        return false;
    }

    const bool has_unsupported_input = input.mamba_init_state_indices.has_value()
                                    || input.mamba_final_state_indices.has_value()
                                    || input.pixel_values.has_value()
                                    || input.image_bound.has_value()
                                    || input.tgt_sizes.has_value()
                                    || input.image_grid_thw.has_value()
                                    || input.image_req_ids.has_value()
                                    || input.visual_token_ranges.has_value()
                                    || input.target_hidden_states.has_value()
                                    || input.sample_all_positions;
    return !has_unsupported_input
        && tensor_values_equal<int64_t>(
            input.position_ids,
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
        && tensor_values_equal<int32_t>(
            input.past_sequence_lengths, {0})
        && tensor_values_equal<int32_t>(
            input.total_sequence_lengths, {10})
        && tensor_values_equal<int32_t>(
            input.input_offsets, {0, 10})
        && tensor_values_equal<int32_t>(
            input.cu_seqlens, {0, 10})
        && tensor_values_equal<int32_t>(
            input.block_tables, {0})
        && tensor_values_equal<int64_t>(
            input.slot_mapping,
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
}

bool supports_reviewed_short_decode_graph(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,
    infinilm::runtime::Device::Type device_type,
    const PagedGraphConfig &graph_config) {
    graph_config.validate();
    if (model_config == nullptr) {
        return false;
    }

    const size_t hidden_size = model_config->get_or<size_t>("hidden_size", 0);
    const size_t num_attention_heads = model_config->get_or<size_t>("num_attention_heads", 0);
    const size_t head_dim = model_config->get_or<size_t>(
        "head_dim",
        num_attention_heads == 0 ? 0 : hidden_size / num_attention_heads);
    const auto dtype = model_config->get_dtype();
    const auto quant_scheme = model_config->get_quant_scheme();
    const auto kv_quant_scheme = model_config->get_kv_quant_scheme();

    if (device_type != infinilm::runtime::Device::Type::kNvidia
        || graph_config.attention_backend != backends::AttentionBackend::FLASH_ATTN
        || hidden_size != 4096
        || num_attention_heads != 32
        || model_config->get_or<size_t>("num_key_value_heads", 0) != 2
        || head_dim != 128
        || paged_config.block_size() != kShortDecodeBlockSize
        || paged_config.num_blocks() < kShortDecodeBlockTableWidth) {
        return false;
    }
    const auto model_type = model_config->get_or<std::string>("model_type", "");
    return model_type == "internlm3"
        || (model_type == "chatglm"
            && paged_config.num_blocks() == 512
            && graph_config.tp_size == 1
            && model_config->get_or<size_t>("num_hidden_layers", 0) == 28
            && model_config->get_or<size_t>("position_id_axes", 1) == 1
            && dtype == infinilm::runtime::DataType::kFloat16
            && quant_scheme == quantization::QuantScheme::NONE
            && kv_quant_scheme == quantization::KVQuantAlgo::NONE);
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

bool uses_reviewed_short_decode_graph(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,
    infinilm::runtime::Device::Type device_type,
    const PagedGraphConfig &graph_config) {
    return supports_reviewed_short_decode_graph(
        paged_config, model_config, device_type, graph_config);
}

FinitePagedDecodeProfile make_finite_paged_decode_profile(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,bool ordinary_kv_prepare,
    infinilm::runtime::Device::Type device_type,const PagedGraphConfig &graph_config) {
    graph_config.validate();
    if(model_config==nullptr) return {};
    const size_t num_heads=model_config->get_or<size_t>("num_attention_heads",0);
    const size_t hidden=model_config->get_or<size_t>("hidden_size",0);
    const size_t head_dim=model_config->get_or<size_t>(
        "head_dim",num_heads==0?0:hidden/num_heads);
    const auto dtype=model_config->get_dtype();
    const bool reviewed_tp2=graph_config.tp_size==2 && graph_config.pp_size==1 &&
        reviewed_qwen3_tp2_finite_decode(
        graph_config.tp_size,graph_config.pp_size,
        model_config->get_or<std::string>("model_type",""),hidden,num_heads,
        model_config->get_or<size_t>("num_key_value_heads",0),
        model_config->get_or<size_t>("num_hidden_layers",0),
        model_config->get_or<size_t>("position_id_axes",1),
        dtype==infinilm::runtime::DataType::kBFloat16);
    return {
        device_type==infinilm::runtime::Device::Type::kNvidia,
        graph_config.attention_backend==backends::AttentionBackend::FLASH_ATTN ||
            graph_config.attention_backend==backends::AttentionBackend::PAGED_ATTN,
        graph_config.tp_size==1 || reviewed_tp2,
        ordinary_kv_prepare,
        model_config->get_quant_scheme()==quantization::QuantScheme::NONE &&
            model_config->get_kv_quant_scheme()==quantization::KVQuantAlgo::NONE,
        dtype==infinilm::runtime::DataType::kFloat16 || dtype==infinilm::runtime::DataType::kBFloat16,
        uses_reviewed_short_decode_graph(paged_config,model_config,device_type,graph_config),
        head_dim,paged_config.block_size(),paged_config.num_blocks()
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
        compiled_baichuan_prefill_b1_s10_.reset();
        compiled_map_decode_.clear();
        block_tables_holder_.reset();
        short_block_tables_holder_.reset();

        const size_t nblocks = paged_config->num_blocks();
        auto &forward_context = forward_context_;
        const bool has_mamba_state = has_mamba_cache(forward_context);

        const auto &model_config = model_->get_model_config();
        if (decode_batch_sizes_.empty() || !model_->supports_kv_only_prepare()
            || !model_config
            || model_config->get_quant_scheme() != quantization::QuantScheme::NONE
            || model_config->get_kv_quant_scheme() != quantization::KVQuantAlgo::NONE
            || (model_config->get_dtype() != infinilm::runtime::DataType::kFloat16
                && model_config->get_dtype() != infinilm::runtime::DataType::kBFloat16)) {
            spdlog::warn("PagedCompiler: isolated prepare is not qualified for this model/state configuration");
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
        const bool prepare_fixed_prefill = supports_baichuan_fixed_prefill_graph(
            *paged_config, model_config.get(), has_mamba_state,
            runtime_.device().type(), graph_config_);
        if (prepare_fixed_prefill) {
            auto prefill_views = paged_prepare_views(
                runtime_, forward_context, nblocks, paged_config->block_size(),
                1, kBaichuanFixedPrefillSequenceLength);
            write_views.insert(write_views.end(), prefill_views.begin(), prefill_views.end());
        }
        PrepareStateGuard prepare_state{runtime_, forward_context, std::move(write_views)};
        decltype(compiled_map_decode_) candidate_decode;
        decltype(compiled_short_decode_b1_) candidate_short_decode;
        decltype(compiled_baichuan_prefill_b1_s10_) candidate_prefill;
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

        auto make_baichuan_fixed_prefill_input = [&]() {
            InfinilmModel::Input input;
            input.input_ids = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillBatchSize,
                 kBaichuanFixedPrefillSequenceLength},
                infinilm::runtime::DataType::kInt64);
            input.position_ids = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillSequenceLength},
                infinilm::runtime::DataType::kInt64);
            input.past_sequence_lengths = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillBatchSize},
                infinilm::runtime::DataType::kInt32);
            input.total_sequence_lengths = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillBatchSize},
                infinilm::runtime::DataType::kInt32);
            input.input_offsets = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillBatchSize + 1},
                infinilm::runtime::DataType::kInt32);
            input.cu_seqlens = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillBatchSize + 1},
                infinilm::runtime::DataType::kInt32);
            input.block_tables = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillBatchSize, 1},
                infinilm::runtime::DataType::kInt32);
            input.slot_mapping = infinilm::runtime::Tensor::empty(
                runtime_,
                {kBaichuanFixedPrefillSequenceLength},
                infinilm::runtime::DataType::kInt64);

            const std::vector<int64_t> position_ids{
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            const std::vector<int32_t> past_sequence_lengths{0};
            const std::vector<int32_t> total_sequence_lengths{
                static_cast<int32_t>(kBaichuanFixedPrefillSequenceLength)};
            const std::vector<int32_t> packed_offsets{
                0,
                static_cast<int32_t>(kBaichuanFixedPrefillSequenceLength)};
            const std::vector<int32_t> block_tables{0};
            const std::vector<int64_t> slot_mapping{
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
            runtime_.setDeviceMemory(input.input_ids.value()->data(), 0, input.input_ids.value()->nbytes());
            runtime_.memcpyH2D(
                input.position_ids.value()->data(),
                position_ids.data(),
                position_ids.size() * sizeof(int64_t),
                false);
            runtime_.memcpyH2D(
                input.past_sequence_lengths.value()->data(),
                past_sequence_lengths.data(),
                past_sequence_lengths.size() * sizeof(int32_t),
                false);
            runtime_.memcpyH2D(
                input.total_sequence_lengths.value()->data(),
                total_sequence_lengths.data(),
                total_sequence_lengths.size() * sizeof(int32_t),
                false);
            runtime_.memcpyH2D(
                input.input_offsets.value()->data(),
                packed_offsets.data(),
                packed_offsets.size() * sizeof(int32_t),
                false);
            runtime_.memcpyH2D(
                input.cu_seqlens.value()->data(),
                packed_offsets.data(),
                packed_offsets.size() * sizeof(int32_t),
                false);
            runtime_.memcpyH2D(
                input.block_tables.value()->data(),
                block_tables.data(),
                block_tables.size() * sizeof(int32_t),
                false);
            runtime_.memcpyH2D(
                input.slot_mapping.value()->data(),
                slot_mapping.data(),
                slot_mapping.size() * sizeof(int64_t),
                false);

            forward_context.attn_metadata = {
                input.past_sequence_lengths,
                input.total_sequence_lengths,
                input.input_offsets,
                input.cu_seqlens,
                input.block_tables,
                input.slot_mapping,
                kBaichuanFixedPrefillSequenceLength,
                kBaichuanFixedPrefillSequenceLength,
            };
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

            barrier_->wait();
            (void)model_->forward(input, forward_context_);
            runtime_.syncStream();
            runtime_.syncStream();
            GraphRecordingGuard recording{runtime_};
            auto output = model_->forward(input, forward_context_);
            auto graph = recording.finish();
            barrier_->wait();

            return make_compiled_result(
                std::move(input), std::move(graph), output);
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
            // The graph report has no memory field. Count unique allocator
            // blocks pinned by captures and retain a device-free waterline;
            // neither quantity purports to include all driver allocations.
            infini::rt::DeviceGuard device_guard{
                runtime_.device().type(), runtime_.device().index(), true};
            size_t free_before = 0, total_bytes = 0;
            INFINILM_CHECK_ERROR(infini::rt::runtime::MemGetInfo(
                &free_before, &total_bytes));
            if (free_before < kBoundedDecodeFreeWaterlineBytes) {
                throw std::runtime_error("paged bucket prepare lacks device free-memory reserve");
            }
            const size_t pinned_before = runtime_.graphPinnedMemoryBytes();
            size_t added_plans = 0;
            for (size_t b : {size_t{1}, size_t{2}}) {
                if (b > max_batch_size) continue;
                for (size_t bound : {size_t{256}, size_t{512}}) {
                    if (++added_plans > kBoundedDecodeMaxPlans) {
                        throw std::logic_error("paged bucket plan count exceeds four");
                    }
                    candidate_decode.emplace(
                        std::make_pair(b, bound),
                        capture_decode(b, bound / kBoundedDecodePageSize,
                                       candidate_holder, bound));
                }
            }
            size_t free_after = 0;
            INFINILM_CHECK_ERROR(infini::rt::runtime::MemGetInfo(
                &free_after, &total_bytes));
            const size_t pinned_after = runtime_.graphPinnedMemoryBytes();
            if (pinned_after < pinned_before
                || pinned_after - pinned_before > kBoundedDecodePinnedBudgetBytes
                || free_after < kBoundedDecodeFreeWaterlineBytes) {
                throw std::runtime_error("paged bucket prepare exceeds pinned/free-memory budget");
            }
            spdlog::info("PagedCompiler: bounded decode plans={}, pinned_delta_bytes={}, device_free_before={}, device_free_after={}",
                         added_plans, pinned_after - pinned_before,
                         free_before, free_after);
        }

        if (uses_reviewed_short_decode_graph(
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

        if (prepare_fixed_prefill) {
            auto input = make_baichuan_fixed_prefill_input();

            if (std::getenv("INFINICORE_GRAPH_DEBUG") != nullptr) {
                spdlog::info(
                    "fixed Baichuan prefill graph compile: rank={}, batch=1, seq=10",
                    graph_config_.tp_rank);
            }
            barrier_->wait();
            (void)model_->forward(input, forward_context_);
            runtime_.syncStream();
            runtime_.syncStream();
            GraphRecordingGuard recording{runtime_};
            auto output = model_->forward(input, forward_context_);
            auto graph = recording.finish();
            barrier_->wait();

            candidate_prefill.emplace(
                make_compiled_result(
                    std::move(input), std::move(graph), output));
        }
        // No graph becomes available until every touched slot and metadata
        // reference has been restored. Failure destroys candidates, not a
        // partially published collection of plans.
        prepare_state.restore();
        compiled_map_decode_.swap(candidate_decode);
        compiled_short_decode_b1_.swap(candidate_short_decode);
        compiled_baichuan_prefill_b1_s10_.swap(candidate_prefill);
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

    const bool use_baichuan_fixed_prefill_graph = compiled_baichuan_prefill_b1_s10_.has_value()
                                               && is_exact_baichuan_fixed_prefill_input(input);
    CompiledResult *selected_result = nullptr;
    bool use_short_decode_graph = false;
    size_t required_pages = 0;
    if (use_baichuan_fixed_prefill_graph) {
        selected_result = &compiled_baichuan_prefill_b1_s10_.value();
    } else {
        // Every other compiled paged graph is decode-only.
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
        || (use_baichuan_fixed_prefill_graph
            && !required_tensors_compatible(
                graph_input.past_sequence_lengths,
                input.past_sequence_lengths))
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

    if (use_baichuan_fixed_prefill_graph
        && std::getenv("INFINICORE_GRAPH_DEBUG") != nullptr) {
        spdlog::info(
            "fixed Baichuan prefill graph hit: rank={}, batch=1, seq=10",
            graph_config_.tp_rank);
    }

    graph_input.input_ids.value()->copy_from(runtime_, input.input_ids.value());
    graph_input.position_ids.value()->copy_from(runtime_, input.position_ids.value());
    if (use_baichuan_fixed_prefill_graph) {
        graph_input.past_sequence_lengths.value()->copy_from(runtime_,
            input.past_sequence_lengths.value());
    }
    graph_input.total_sequence_lengths.value()->copy_from(runtime_, input.total_sequence_lengths.value());
    graph_input.input_offsets.value()->copy_from(runtime_, input.input_offsets.value());
    graph_input.cu_seqlens.value()->copy_from(runtime_, input.cu_seqlens.value());

    auto &graph_block_tables = graph_input.block_tables.value();
    if (use_baichuan_fixed_prefill_graph) {
        graph_block_tables->copy_from(runtime_, runtime_block_tables);
    } else if (use_short_decode_graph) {
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

    const bool graph_has_mamba_indices = graph_input.mamba_init_state_indices.has_value()
                                      && graph_input.mamba_final_state_indices.has_value();
            if (graph_has_mamba_indices) {
        graph_input.mamba_init_state_indices.value()->copy_from(runtime_,
                    input.mamba_init_state_indices.value());
        graph_input.mamba_final_state_indices.value()->copy_from(runtime_,
                    input.mamba_final_state_indices.value());
            }

    auto graph = std::get<0>(selected_result->compiled);
    const auto &compiled_output = std::get<1>(selected_result->compiled);
    auto shared_output = resume_output(*compiled_output);

            return std::make_tuple(graph, shared_output);
        }

} // namespace infinilm::engine
