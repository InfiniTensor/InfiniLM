#include "paged_compiler.hpp"
#include "../../global_state/global_state.hpp"
#include "../../utils.hpp"
#include "infinicore/ops/distributed/allreduce.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
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
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        const bool pp_graph = rank_info.pp_size > 1
                           && std::getenv("INFINILM_EXPERIMENTAL_PP_GRAPH") != nullptr;
        auto sync_pp_stages = [&] {
            if (!pp_graph) {
                return;
            }
            auto marker = infinicore::Tensor::empty(
                {1}, infinicore::DataType::I32, infinicore::context::getDevice());
            set_zeros(marker);
            infinicore::op::distributed::allreduce_(
                marker, marker, INFINICCL_SUM, rank_info.world_comm);
            infinicore::context::syncStream();
        };
        size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
        auto &forward_context = infinilm::global_state::get_forward_context();
        const bool has_mamba_state = has_mamba_cache(forward_context);
        size_t block_size = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->block_size();

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
            input.request_ids = infinicore::Tensor::empty(
                {b}, infinicore::DataType::I32, infinicore::context::getDevice());
            infinicore::context::memcpyH2D(
                input.request_ids.value()->data(), input_offsets_vec.data(),
                b * sizeof(int32_t), false);
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
                input.request_ids,
                input.cu_seqlens,
                input.block_tables,
                input.slot_mapping,
                0,
                0,
                static_cast<int64_t>(nblocks * block_size),
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
            sync_pp_stages();
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

            sync_pp_stages();
            barrier_->wait();
            (void)model_->forward(input);
            infinicore::context::syncStream();
            // Capture must not start with stale Marlin locks from previous
            // warmup/capture attempts. This reset is intentionally outside
            // graph capture; the current implementation still pays a memset
            // before every graph replay in get_compiled().
            model_->reset_runtime_state();
            infinicore::context::syncStream();
            sync_pp_stages();
            infinicore::context::startGraphRecording();
            auto output = model_->forward(input);
            auto graph = infinicore::context::stopGraphRecording();
            barrier_->wait();

            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{
                    output.logits
                        ? infinicore::graph::GraphTensor(output.logits)
                        : infinicore::Tensor()});

            compiled_map_decode_[b] = CompiledResult{std::move(input), std::make_tuple(graph, shared_output)};
        }
        sync_pp_stages();
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

            if (batch_size == 2
                && std::getenv("INFINILM_EXPERIMENTAL_PP_GRAPH_VERIFY_TABLE") != nullptr) {
                if (input.block_tables.value()->device().getType()
                    != infinicore::Device::Type::CPU) {
                    throw std::runtime_error("PagedCompiler table verification requires CPU input");
                }
                infinicore::context::syncStream();
                std::vector<int32_t> observed(batch_size * compiled_block_per_req);
                infinicore::context::memcpyD2H(
                    observed.data(), graph_block_tables->data(),
                    observed.size() * sizeof(int32_t));
                const auto *expected = reinterpret_cast<const int32_t *>(
                    input.block_tables.value()->data());
                for (size_t row = 0; row < batch_size; ++row) {
                    for (size_t col = 0; col < compiled_block_per_req; ++col) {
                        const int32_t value = col < block_per_req
                                                ? expected[row * block_per_req + col]
                                                : -1;
                        if (observed[row * compiled_block_per_req + col] != value) {
                            throw std::runtime_error(
                                "PagedCompiler graph block table mismatch at row "
                                + std::to_string(row) + ", column "
                                + std::to_string(col));
                        }
                    }
                }
                auto verify_i32 = [](const char *name,
                                     const infinicore::Tensor &graph_tensor,
                                     const infinicore::Tensor &runtime_tensor) {
                    std::vector<int32_t> actual(graph_tensor->numel());
                    infinicore::context::memcpyD2H(
                        actual.data(), graph_tensor->data(),
                        actual.size() * sizeof(int32_t));
                    const auto *wanted = reinterpret_cast<const int32_t *>(runtime_tensor->data());
                    for (size_t i = 0; i < actual.size(); ++i) {
                        if (actual[i] != wanted[i]) {
                            throw std::runtime_error(
                                std::string("PagedCompiler graph ") + name
                                + " mismatch at element " + std::to_string(i));
                        }
                    }
                };
                auto verify_i64 = [](const char *name,
                                     const infinicore::Tensor &graph_tensor,
                                     const infinicore::Tensor &runtime_tensor) {
                    std::vector<int64_t> actual(graph_tensor->numel());
                    infinicore::context::memcpyD2H(
                        actual.data(), graph_tensor->data(),
                        actual.size() * sizeof(int64_t));
                    const auto *wanted = reinterpret_cast<const int64_t *>(runtime_tensor->data());
                    for (size_t i = 0; i < actual.size(); ++i) {
                        if (actual[i] != wanted[i]) {
                            throw std::runtime_error(
                                std::string("PagedCompiler graph ") + name
                                + " mismatch at element " + std::to_string(i));
                        }
                    }
                };
                verify_i64("input_ids", graph_input.input_ids.value(), input.input_ids.value());
                verify_i64("position_ids", graph_input.position_ids.value(), input.position_ids.value());
                verify_i32("total_sequence_lengths", graph_input.total_sequence_lengths.value(), input.total_sequence_lengths.value());
                verify_i32("input_offsets", graph_input.input_offsets.value(), input.input_offsets.value());
                verify_i32("cu_seqlens", graph_input.cu_seqlens.value(), input.cu_seqlens.value());
                verify_i64("slot_mapping", graph_input.slot_mapping.value(), input.slot_mapping.value());
            }

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
            const auto &compiled_logits = std::get<1>(result->second.compiled)->logits;
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                new InfinilmModel::Output{
                    compiled_logits ? compiled_logits->resume_from_blob_()
                                    : infinicore::Tensor()});

            return std::make_tuple(graph, shared_output);
        }
    } else {
        return {nullptr, nullptr};
    }
}

} // namespace infinilm::engine
