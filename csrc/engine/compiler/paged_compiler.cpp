#include "paged_compiler.hpp"
#include "backends/operators/operators.hpp"
#include "../../global_state/global_state.hpp"

#include <cstring>

namespace {
// Todo: replace with Tensor::zeros when it is available
inline void set_zeros(infinicore::Tensor &tensor) {
    std::vector<uint8_t> zeros(tensor->nbytes(), 0);
    infinicore::context::memcpyH2D(tensor->data(), zeros.data(), tensor->nbytes(), false);
}

inline void set_zeros_cpu(infinicore::Tensor &tensor) {
    std::memset(tensor->data(), 0, tensor->nbytes());
}

inline void set_minus_one_cpu(infinicore::Tensor &tensor) {
    std::memset(tensor->data(), 0xFF, tensor->nbytes());
}

inline void copy_cpu_tensor_bytes(infinicore::Tensor &dst, const infinicore::Tensor &src) {
    if (dst->device().getType() != infinicore::Device::Type::CPU
        || src->device().getType() != infinicore::Device::Type::CPU
        || dst->shape() != src->shape()
        || dst->dtype() != src->dtype()
        || !dst->is_contiguous()
        || !src->is_contiguous()) {
        throw std::runtime_error("PagedCompiler: expected matching contiguous CPU tensors for host metadata copy.");
    }
    std::memcpy(dst->data(), src->data(), dst->nbytes());
}

inline void copy_block_tables_to_host(infinicore::Tensor &dst,
                                      const infinicore::Tensor &src,
                                      size_t block_per_req) {
    if (dst->device().getType() != infinicore::Device::Type::CPU
        || src->device().getType() != infinicore::Device::Type::CPU
        || dst->dtype() != infinicore::DataType::I32
        || src->dtype() != infinicore::DataType::I32
        || dst->ndim() != 2
        || src->ndim() != 2
        || dst->size(0) != src->size(0)
        || src->size(1) != block_per_req
        || dst->size(1) < block_per_req
        || !dst->is_contiguous()
        || !src->is_contiguous()) {
        throw std::runtime_error("PagedCompiler: expected matching contiguous CPU block tables.");
    }

    const auto rows = dst->size(0);
    const auto dst_cols = dst->size(1);
    const auto *src_data = reinterpret_cast<const int32_t *>(src->data());
    auto *dst_data = reinterpret_cast<int32_t *>(dst->data());
    for (size_t row = 0; row < rows; ++row) {
        std::memcpy(dst_data + row * dst_cols,
                    src_data + row * block_per_req,
                    block_per_req * sizeof(int32_t));
    }
}

} // namespace
namespace infinilm::engine {
PagedCompiler::PagedCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    if (infinicore::context::getDevice().getType() == infinicore::Device::Type::ASCEND) {
        decode_batch_sizes_.push_back(1);
        return;
    }

    for (size_t b = 1; b < 64; ++b) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 64; b < 128; b += 16) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 128; b < 256; b += 32) {
        decode_batch_sizes_.push_back(b);
    }
    for (size_t b = 256; b <= 512; b += 64) {
        decode_batch_sizes_.push_back(b);
    }
}

void PagedCompiler::compile() {
    if (model_->get_cache_config() != nullptr && dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        size_t nblocks = dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())->num_blocks();
        size_t max_batch_size = *std::max_element(decode_batch_sizes_.begin(), decode_batch_sizes_.end());
        compiled_map_decode_.clear();
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks * max_batch_size}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);
        for (size_t b : decode_batch_sizes_) {
            InfinilmModel::Input input;
            input.input_ids = infinicore::Tensor::empty({1, b}, infinicore::DataType::I64, infinicore::context::getDevice());
            input.position_ids = infinicore::Tensor::empty({b}, infinicore::DataType::I64, infinicore::context::getDevice());
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

            auto device = infinicore::context::getDevice();
            auto total_sequence_lengths_host = infinicore::Tensor::empty({b}, infinicore::DataType::I32, infinicore::Device::cpu());
            std::memcpy(total_sequence_lengths_host->data(), total_sequence_lengths_vec.data(), b * sizeof(int32_t));
            auto block_tables_host = infinicore::Tensor::empty({b, block_per_req}, infinicore::DataType::I32, infinicore::Device::cpu());
            set_zeros_cpu(block_tables_host);
            infinicore::context::setDevice(device);

            // Attention reads attn_metadata from thread-local forward context.
            infinilm::global_state::get_forward_context().attn_metadata = {
                input.past_sequence_lengths,
                input.total_sequence_lengths,
                input.input_offsets,
                input.cu_seqlens,
                input.block_tables,
                input.slot_mapping,
                total_sequence_lengths_host,
                block_tables_host,
            };

            barrier_->wait();
            backends::ops::begin_graph_task_capture();
            try {
                infinicore::context::startGraphRecording();
                auto output = model_->forward(input);
                // Per-op fence locks the first eager-warmup pass inside instantiate
                // step-by-step across ranks. Fixes DTK 2604 HSA-loader freeze race
                // (divergent concurrent freezes wedge BLIT-DMA); same-kernel
                // concurrent freezes are fine, so syncing every op makes it safe.
                // Visible on 70B tp=4 graph (80-layer op_list); 8B tp=4 graph
                // happened to fit within the loader's tolerance window.
                auto graph = infinicore::context::stopGraphRecording(
                    [this]() { barrier_->wait(); });
                auto graph_task_updates = backends::ops::end_graph_task_capture();
                barrier_->wait();

                auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                    new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

                compiled_map_decode_[b] = CompiledResult{
                    std::move(input),
                    total_sequence_lengths_host,
                    block_tables_host,
                    std::move(graph_task_updates),
                    std::make_tuple(graph, shared_output)};
            } catch (...) {
                (void)backends::ops::end_graph_task_capture();
                throw;
            }
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
            auto &graph_input = result->second.input;

            graph_input.input_ids.value()->copy_from(input.input_ids.value());
            graph_input.position_ids.value()->copy_from(input.position_ids.value());
            graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
            graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
            graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());
            copy_cpu_tensor_bytes(result->second.total_sequence_lengths_host, input.total_sequence_lengths.value());

            const size_t compiled_block_per_req = graph_input.block_tables.value()->size(1);
            if (block_per_req > compiled_block_per_req) {
                // Runtime width exceeds compiled graph slot; fall back to eager path.
                return {nullptr, nullptr};
            }

            auto &host_block_tables = result->second.block_tables_host;
            set_minus_one_cpu(host_block_tables);
            copy_block_tables_to_host(host_block_tables, input.block_tables.value(), block_per_req);
            graph_input.block_tables.value()->copy_from(host_block_tables);
            graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());
            backends::ops::update_graph_tasks(result->second.graph_task_updates);

            auto graph = std::get<0>(result->second.compiled);
            auto shared_output = std::shared_ptr<InfinilmModel::Output>(new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});

            return std::make_tuple(graph, shared_output);
        }
    } else {
        return {nullptr, nullptr};
    }
}

} // namespace infinilm::engine
