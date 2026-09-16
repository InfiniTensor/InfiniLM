#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <numeric>
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
            const size_t block_per_req = nblocks;
            input.block_tables = block_tables_holder_->as_strided({b, block_per_req}, {(ptrdiff_t)block_per_req, 1});
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
            };
            return input;
        };

        {
            const size_t warmup_batch_size = std::min(max_batch_size, static_cast<size_t>(64));
            auto input = make_decode_input(warmup_batch_size);
            model_->forward(input);
            infinicore::context::syncStream();
            // Warmup runs the eager Marlin path and may leave per-layer lock
            // workspaces dirty. Reset before CUDA graph capture so capture
            // starts from the same all-zero lock state as normal execution.
            model_->reset_runtime_state();
            infinicore::context::syncStream();
        }

        for (size_t b : decode_batch_sizes_) {
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
            infinicore::context::startGraphRecording();
            auto output = model_->forward(input);
            auto graph = infinicore::context::stopGraphRecording();
            barrier_->wait();

            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

            compiled_map_decode_[b] = CompiledResult{std::move(input), std::make_tuple(graph, shared_output)};
        }
    }
    compile_prefill();
}

void PagedCompiler::compile_prefill() {
    compiled_map_prefill_.clear();
    prefill_chunk_size_ = 0;
    const char *size_env = std::getenv("INFINILM_PREFILL_GRAPH_CHUNK_SIZE");
    if (size_env == nullptr) {
        return;
    }
    // This opt-in experiment deliberately has a smaller support scope than
    // the existing Decode compiler. Do not silently capture unsupported paths.
    char *end = nullptr;
    const auto chunk = std::strtoul(size_env, &end, 10);
    const auto *cache = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
    const auto &config = infinilm::global_state::get_infinilm_config();
    auto &ctx = infinilm::global_state::get_forward_context();
    if (end == size_env || *end != '\0' || chunk < 2 || chunk > 4096
        || cache == nullptr || chunk > cache->num_blocks() * cache->block_size()
        || config.attention_backend != infinilm::backends::AttentionBackend::FLASH_ATTN
        || config.use_mla || has_mamba_cache(ctx)
        || infinilm::global_state::get_tensor_model_parallel_world_size() != 1
        || infinicore::context::getDevice().getType() != infinicore::Device::Type::METAX) {
        throw std::invalid_argument("Experimental Prefill graphs require C500/MetaX, TP1, flash-attn, ordinary paged KV and chunk size 2..4096 within cache capacity");
    }
    prefill_chunk_size_ = chunk;
    // Flash Attention's scalar maximum is fixed during capture. The actual
    // KV length remains a device tensor and can grow across continuation chunks.
    prefill_max_sequence_length_ = cache->num_blocks() * cache->block_size();
    auto make_tensor = [](const std::vector<size_t> &shape, infinicore::DataType dtype) {
        return infinicore::Tensor::empty(shape, dtype, infinicore::context::getDevice());
    };
    for (bool intermediate : {true, false}) {
        InfinilmModel::Input input;
        input.prefill_only = intermediate;
        input.input_ids = make_tensor({1, chunk}, infinicore::DataType::I64);
        input.position_ids = make_tensor({chunk}, infinicore::DataType::I64);
        input.total_sequence_lengths = make_tensor({1}, infinicore::DataType::I32);
        input.input_offsets = make_tensor({2}, infinicore::DataType::I32);
        input.cu_seqlens = make_tensor({2}, infinicore::DataType::I32);
        input.block_tables = make_tensor({1, cache->num_blocks()}, infinicore::DataType::I32);
        input.slot_mapping = make_tensor({chunk}, infinicore::DataType::I64);
        set_zeros(input.input_ids.value());
        std::vector<int64_t> positions(chunk);
        std::iota(positions.begin(), positions.end(), 0);
        std::vector<int32_t> pages(cache->num_blocks());
        std::iota(pages.begin(), pages.end(), 0);
        const std::vector<int32_t> offsets{0, static_cast<int32_t>(chunk)};
        auto upload = [](infinicore::Tensor dst, const auto &values) {
            infinicore::context::memcpyH2D(dst->data(), values.data(), values.size() * sizeof(values[0]), false);
        };
        upload(input.position_ids.value(), positions);
        upload(input.slot_mapping.value(), positions);
        upload(input.block_tables.value(), pages);
        upload(input.input_offsets.value(), offsets);
        upload(input.cu_seqlens.value(), offsets);
        upload(input.total_sequence_lengths.value(), std::vector<int32_t>{static_cast<int32_t>(chunk)});
        ctx.attn_metadata = {input.past_sequence_lengths, input.total_sequence_lengths,
                             input.input_offsets, input.cu_seqlens, input.block_tables,
                             input.slot_mapping, chunk, prefill_max_sequence_length_};
        (void)model_->forward(input);
        infinicore::context::syncStream();
        model_->reset_runtime_state();
        infinicore::context::syncStream();
        infinicore::context::startGraphRecording();
        auto output = model_->forward(input);
        auto graph = infinicore::context::stopGraphRecording();
        auto saved = std::make_shared<InfinilmModel::Output>();
        if (output.logits) {
            saved->logits = infinicore::graph::GraphTensor(output.logits);
        }
        if (output.hidden_states) {
            saved->hidden_states = infinicore::graph::GraphTensor(output.hidden_states);
        }
        compiled_map_prefill_[intermediate] = CompiledResult{std::move(input), {graph, saved}};
    }
}

PagedCompiler::Compiled PagedCompiler::get_compiled_prefill(const InfinilmModel::Input &input) {
    if (prefill_chunk_size_ == 0 || input.input_ids.value()->numel() != prefill_chunk_size_
        || input.block_tables.value()->size(0) != 1 || input.sample_all_positions
        || input.mamba_init_state_indices.has_value() || input.pixel_values.has_value()) {
        return {nullptr, nullptr};
    }
    auto &entry = compiled_map_prefill_.at(input.prefill_only);
    auto &target = entry.input;
    const auto &lengths = input.total_sequence_lengths.value();
    if (lengths->device().getType() != infinicore::Device::Type::CPU
        || lengths->dtype() != infinicore::DataType::I32 || lengths->numel() != 1) {
        throw std::invalid_argument("Prefill graph replay requires CPU int32 sequence lengths");
    }
    const auto length = *reinterpret_cast<const int32_t *>(lengths->data());
    const size_t width = input.block_tables.value()->size(1);
    if (length < static_cast<int32_t>(prefill_chunk_size_)
        || static_cast<size_t>(length) > prefill_max_sequence_length_
        || width > target.block_tables.value()->size(1)
        || input.position_ids.value()->shape() != target.position_ids.value()->shape()) {
        return {nullptr, nullptr};
    }
    target.input_ids.value()->copy_from(input.input_ids.value());
    target.position_ids.value()->copy_from(input.position_ids.value());
    target.total_sequence_lengths.value()->copy_from(lengths);
    target.input_offsets.value()->copy_from(input.input_offsets.value());
    target.cu_seqlens.value()->copy_from(input.cu_seqlens.value());
    target.slot_mapping.value()->copy_from(input.slot_mapping.value());
    set_minus_one_device_async(target.block_tables.value());
    target.block_tables.value()->narrow({{1, 0, width}})->copy_from(input.block_tables.value());
    model_->reset_runtime_state();
    auto saved = std::get<1>(entry.compiled);
    auto output = std::make_shared<InfinilmModel::Output>();
    if (saved->logits) {
        output->logits = saved->logits->resume_from_blob_();
    }
    return {std::get<0>(entry.compiled), output};
}

PagedCompiler::Compiled PagedCompiler::get_compiled(const InfinilmModel::Input &input) {
    if (prefill_chunk_size_ != 0) {
        auto result = get_compiled_prefill(input);
        if (std::get<0>(result)) {
            return result;
        }
    }
    if (input.prefill_only) {
        return {nullptr, nullptr};
    }
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
