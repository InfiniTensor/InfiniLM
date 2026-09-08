#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace infinilm::engine {
namespace {

constexpr size_t kP13ShortDecodeBlockTableWidth = 8;
constexpr size_t kP13ShortDecodeBlockSize = 256;
constexpr size_t kP13ShortDecodeMaxSequenceLength =
    kP13ShortDecodeBlockTableWidth * kP13ShortDecodeBlockSize;

constexpr char kBaichuanFixedPrefillGraphEnv[] =
    "INFINILM_ENABLE_BAICHUAN_PREFILL_GRAPH";
constexpr size_t kBaichuanFixedPrefillBatchSize = 1;
constexpr size_t kBaichuanFixedPrefillSequenceLength = 10;
constexpr size_t kBaichuanFixedPrefillBlockSize = 256;

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

bool env_flag_enabled(const char *name) {
    const char *value = std::getenv(name);
    return value != nullptr && std::string_view(value) == "1";
}

bool supports_baichuan_fixed_prefill_graph(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config,
    bool has_mamba_state) {
    if (!env_flag_enabled(kBaichuanFixedPrefillGraphEnv)
        || model_config == nullptr) {
        return false;
    }

    const size_t hidden_size = model_config->get_or<size_t>("hidden_size", 0);
    const size_t num_attention_heads =
        model_config->get_or<size_t>("num_attention_heads", 0);
    const size_t head_dim = model_config->get_or<size_t>(
        "head_dim",
        num_attention_heads == 0 ? 0 : hidden_size / num_attention_heads);

    return infinicore::context::getDevice().type()
               == infinicore::Device::Type::kNvidia
        && infinilm::global_state::get_infinilm_config().attention_backend
               == backends::AttentionBackend::FLASH_ATTN
        && infinilm::global_state::get_tensor_model_parallel_world_size() == 2
        && paged_config.block_size() == kBaichuanFixedPrefillBlockSize
        && paged_config.num_blocks() == 1
        && paged_config.max_batch_size() == kBaichuanFixedPrefillBatchSize
        && model_config->get_or<std::string>("model_type", "") == "baichuan"
        && hidden_size == 4096
        && model_config->get_or<size_t>("num_hidden_layers", 0) == 32
        && num_attention_heads == 32
        && model_config->get_or<size_t>("num_key_value_heads", 0) == 32
        && head_dim == 128
        && model_config->get_or<size_t>("position_id_axes", 1) == 1
        && model_config->get_quant_scheme()
               == quantization::QuantScheme::NONE
        && model_config->get_kv_quant_scheme()
               == quantization::KVQuantAlgo::NONE
        && !has_mamba_state;
}

bool is_cpu_contiguous_tensor(
    const std::optional<infinicore::Tensor> &tensor,
    infinicore::DataType dtype,
    const std::vector<size_t> &shape) {
    return tensor.has_value()
        && tensor.value()
        && tensor.value()->device().type() == infinicore::Device::Type::kCpu
        && tensor.value()->dtype() == dtype
        && tensor.value()->shape() == shape
        && tensor.value()->is_contiguous();
}

template <typename T>
bool tensor_values_equal(
    const std::optional<infinicore::Tensor> &tensor,
    std::initializer_list<T> expected) {
    const auto *values =
        reinterpret_cast<const T *>(tensor.value()->data());
    return std::equal(expected.begin(), expected.end(), values);
}

bool is_exact_baichuan_fixed_prefill_input(
    const InfinilmModel::Input &input) {
    const bool tensors_match =
        is_cpu_contiguous_tensor(
            input.input_ids,
            infinicore::DataType::kInt64,
            {kBaichuanFixedPrefillBatchSize,
             kBaichuanFixedPrefillSequenceLength})
        && is_cpu_contiguous_tensor(
            input.position_ids,
            infinicore::DataType::kInt64,
            {kBaichuanFixedPrefillSequenceLength})
        && is_cpu_contiguous_tensor(
            input.past_sequence_lengths,
            infinicore::DataType::kInt32,
            {kBaichuanFixedPrefillBatchSize})
        && is_cpu_contiguous_tensor(
            input.total_sequence_lengths,
            infinicore::DataType::kInt32,
            {kBaichuanFixedPrefillBatchSize})
        && is_cpu_contiguous_tensor(
            input.input_offsets,
            infinicore::DataType::kInt32,
            {kBaichuanFixedPrefillBatchSize + 1})
        && is_cpu_contiguous_tensor(
            input.cu_seqlens,
            infinicore::DataType::kInt32,
            {kBaichuanFixedPrefillBatchSize + 1})
        && is_cpu_contiguous_tensor(
            input.block_tables,
            infinicore::DataType::kInt32,
            {kBaichuanFixedPrefillBatchSize, 1})
        && is_cpu_contiguous_tensor(
            input.slot_mapping,
            infinicore::DataType::kInt64,
            {kBaichuanFixedPrefillSequenceLength});
    if (!tensors_match) {
        return false;
    }

    const bool has_unsupported_input =
        input.mamba_init_state_indices.has_value()
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

bool supports_p13_short_decode_graph(
    const cache::PagedKVCacheConfig &paged_config,
    const config::ModelConfig *model_config) {
    if (model_config == nullptr) {
        return false;
    }

    const size_t hidden_size = model_config->get_or<size_t>("hidden_size", 0);
    const size_t num_attention_heads =
        model_config->get_or<size_t>("num_attention_heads", 0);
    const size_t head_dim = model_config->get_or<size_t>(
        "head_dim",
        num_attention_heads == 0 ? 0 : hidden_size / num_attention_heads);

    return infinicore::context::getDevice().type()
               == infinicore::Device::Type::kNvidia
        && infinilm::global_state::get_infinilm_config().attention_backend
               == backends::AttentionBackend::FLASH_ATTN
        && paged_config.block_size() == kP13ShortDecodeBlockSize
        && paged_config.num_blocks() >= kP13ShortDecodeBlockTableWidth
        && model_config->get_or<std::string>("model_type", "") == "internlm3"
        && hidden_size == 4096
        && num_attention_heads == 32
        && model_config->get_or<size_t>("num_key_value_heads", 0) == 2
        && head_dim == 128;
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
    if (paged_config != nullptr) {
        compiled_short_decode_b1_.reset();
        compiled_baichuan_prefill_b1_s10_.reset();
        compiled_map_decode_.clear();
        block_tables_holder_.reset();
        short_block_tables_holder_.reset();

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

        auto make_baichuan_fixed_prefill_input = [&]() {
            InfinilmModel::Input input;
            input.input_ids = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillBatchSize,
                 kBaichuanFixedPrefillSequenceLength},
                infinicore::DataType::kInt64,
                infinicore::context::getDevice());
            input.position_ids = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillSequenceLength},
                infinicore::DataType::kInt64,
                infinicore::context::getDevice());
            input.past_sequence_lengths = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillBatchSize},
                infinicore::DataType::kInt32,
                infinicore::context::getDevice());
            input.total_sequence_lengths = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillBatchSize},
                infinicore::DataType::kInt32,
                infinicore::context::getDevice());
            input.input_offsets = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillBatchSize + 1},
                infinicore::DataType::kInt32,
                infinicore::context::getDevice());
            input.cu_seqlens = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillBatchSize + 1},
                infinicore::DataType::kInt32,
                infinicore::context::getDevice());
            input.block_tables = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillBatchSize, 1},
                infinicore::DataType::kInt32,
                infinicore::context::getDevice());
            input.slot_mapping = infinicore::Tensor::empty(
                {kBaichuanFixedPrefillSequenceLength},
                infinicore::DataType::kInt64,
                infinicore::context::getDevice());

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
            set_zeros(input.input_ids.value());
            infinicore::context::memcpyH2D(
                input.position_ids.value()->data(),
                position_ids.data(),
                position_ids.size() * sizeof(int64_t),
                false);
            infinicore::context::memcpyH2D(
                input.past_sequence_lengths.value()->data(),
                past_sequence_lengths.data(),
                past_sequence_lengths.size() * sizeof(int32_t),
                false);
            infinicore::context::memcpyH2D(
                input.total_sequence_lengths.value()->data(),
                total_sequence_lengths.data(),
                total_sequence_lengths.size() * sizeof(int32_t),
                false);
            infinicore::context::memcpyH2D(
                input.input_offsets.value()->data(),
                packed_offsets.data(),
                packed_offsets.size() * sizeof(int32_t),
                false);
            infinicore::context::memcpyH2D(
                input.cu_seqlens.value()->data(),
                packed_offsets.data(),
                packed_offsets.size() * sizeof(int32_t),
                false);
            infinicore::context::memcpyH2D(
                input.block_tables.value()->data(),
                block_tables.data(),
                block_tables.size() * sizeof(int32_t),
                false);
            infinicore::context::memcpyH2D(
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
            compiled_map_decode_[b] =
                capture_decode(b, nblocks, block_tables_holder_);
        }

        if (supports_p13_short_decode_graph(
                *paged_config, model_->get_model_config().get())) {
            short_block_tables_holder_ = infinicore::Tensor::empty(
                {kP13ShortDecodeBlockTableWidth},
                infinicore::DataType::kInt32,
                infinicore::context::getDevice());
            set_zeros(short_block_tables_holder_);
            compiled_short_decode_b1_.emplace(capture_decode(
                1,
                kP13ShortDecodeBlockTableWidth,
                short_block_tables_holder_));
        }

        if (supports_baichuan_fixed_prefill_graph(
                *paged_config, model_->get_model_config().get(), has_mamba_state)) {
            auto input = make_baichuan_fixed_prefill_input();

            if (std::getenv("INFINICORE_GRAPH_DEBUG") != nullptr) {
                spdlog::info(
                    "fixed Baichuan prefill graph compile: rank={}, batch=1, seq=10",
                    infinilm::global_state::get_tensor_model_parallel_rank());
            }
            barrier_->wait();
            (void)model_->forward(input);
            infinicore::context::syncStream();
            model_->reset_runtime_state();
            infinicore::context::syncStream();
            GraphRecordingGuard recording;
            auto output = model_->forward(input);
            auto graph = recording.finish();
            barrier_->wait();

            compiled_baichuan_prefill_b1_s10_.emplace(
                make_compiled_result(
                    std::move(input), std::move(graph), output));
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

    const bool use_baichuan_fixed_prefill_graph =
        compiled_baichuan_prefill_b1_s10_.has_value()
        && is_exact_baichuan_fixed_prefill_input(input);
    CompiledResult *selected_result = nullptr;
    bool use_short_decode_graph = false;
    size_t required_pages = 0;
    if (use_baichuan_fixed_prefill_graph) {
        selected_result = &compiled_baichuan_prefill_b1_s10_.value();
    } else {
        // Every other compiled paged graph is decode-only.
        if (batch_size != input.input_ids.value()->size(1)) {
            return {nullptr, nullptr};
        }

        auto general_result = compiled_map_decode_.find(batch_size);
        if (general_result == compiled_map_decode_.end()) {
            return {nullptr, nullptr};
        }
        selected_result = &general_result->second;

        if (batch_size == 1 && compiled_short_decode_b1_.has_value()) {
            const auto &total_sequence_lengths = input.total_sequence_lengths;
            const bool valid_short_metadata =
                total_sequence_lengths.has_value()
                && total_sequence_lengths.value()
                && total_sequence_lengths.value()->device().type()
                       == infinicore::Device::Type::kCpu
                && total_sequence_lengths.value()->dtype()
                       == infinicore::DataType::kInt32
                && total_sequence_lengths.value()->ndim() == 1
                && total_sequence_lengths.value()->size(0) == 1
                && total_sequence_lengths.value()->is_contiguous()
                && runtime_block_tables->device().type()
                       == infinicore::Device::Type::kCpu
                && runtime_block_tables->dtype() == infinicore::DataType::kInt32
                && runtime_block_tables->is_contiguous();
            if (valid_short_metadata) {
                const int32_t total_sequence_length =
                    reinterpret_cast<const int32_t *>(
                        total_sequence_lengths.value()->data())[0];
                if (total_sequence_length > 0
                    && static_cast<size_t>(total_sequence_length)
                           <= kP13ShortDecodeMaxSequenceLength) {
                    required_pages =
                        1
                        + (static_cast<size_t>(total_sequence_length) - 1)
                              / paged_config->block_size();
                    if (required_pages <= kP13ShortDecodeBlockTableWidth
                        && block_per_req >= required_pages) {
                        selected_result = &compiled_short_decode_b1_.value();
                        use_short_decode_graph = true;
                    }
                }
            }
        }
    }

    auto &graph_input = selected_result->input;
    const size_t compiled_block_per_req =
        graph_input.block_tables.value()->size(1);
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
            infinilm::global_state::get_tensor_model_parallel_rank());
    }

    graph_input.input_ids.value()->copy_from(input.input_ids.value());
    graph_input.position_ids.value()->copy_from(input.position_ids.value());
    if (use_baichuan_fixed_prefill_graph) {
        graph_input.past_sequence_lengths.value()->copy_from(
            input.past_sequence_lengths.value());
    }
    graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
    graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
    graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());

    auto &graph_block_tables = graph_input.block_tables.value();
    if (use_baichuan_fixed_prefill_graph) {
        graph_block_tables->copy_from(runtime_block_tables);
    } else if (use_short_decode_graph) {
        infinicore::context::setDevice(graph_block_tables->device());
        set_minus_one_device_async(graph_block_tables);
        infinicore::context::memcpyH2D(
            graph_block_tables->data(),
            runtime_block_tables->data(),
            required_pages * sizeof(int32_t),
            false);
    } else {
        // Initialize only the active graph rows to -1, then overwrite the
        // runtime logical region. Avoid clearing the full preallocated holder
        // on every decode token.
        set_minus_one_device_async(graph_block_tables);
        graph_block_tables->narrow({{1, 0, block_per_req}})
            ->copy_from(runtime_block_tables);
    }
    graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());

    const bool graph_has_mamba_indices =
        graph_input.mamba_init_state_indices.has_value()
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
    if (!use_baichuan_fixed_prefill_graph) {
        model_->reset_runtime_state();
    }

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
