#include "chunk_prefill_compiler.hpp"
#include "infinicore/context/context.hpp"


namespace {
inline void set_zeros(infinicore::Tensor &tensor) {
    std::vector<uint8_t> zeros(tensor->nbytes(), 0);
    infinicore::context::memcpyH2D(tensor->data(), zeros.data(), tensor->nbytes(), false);
}
} // namespace

namespace infinilm::engine {

ChunkPrefillCompiler::ChunkPrefillCompiler(const std::shared_ptr<InfinilmModel> &model, RankBarrier *barrier)
    : GraphCompiler(model, barrier) {
    // Enumerate chunk sizes for chunk-prefill
    for (size_t cs : {64, 128, 256, 512, 1024, 2048}) {
        chunk_sizes_.push_back(cs);
    }
    // Enumerate batch sizes for prefill (typically smaller than decode)
    for (size_t b = 1; b < 32; b++) {
        prefill_batch_sizes_.push_back(b);
    }
    for (size_t b = 32; b < 64; b += 8) {
        prefill_batch_sizes_.push_back(b);
    }
    for (size_t b = 64; b < 128; b += 16) {
        prefill_batch_sizes_.push_back(b);
    }
    for (size_t b = 128; b < 256; b += 32) {
        prefill_batch_sizes_.push_back(b);
    }
    for (size_t b = 256; b <= 512; b += 64) {
        prefill_batch_sizes_.push_back(b);
    }
}

void ChunkPrefillCompiler::compile() {
    if (model_->get_cache_config() != nullptr &&
        dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {

        const auto *paged_config =
            dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config());
        size_t nblocks = paged_config->num_blocks();

        compiled_map_prefill_.clear();

        // Max total tokens to avoid OOM during graph recording
        constexpr size_t MAX_TOTAL_TOKENS = 4096;

        // Pre-allocate a shared block_tables_holder for the largest (batch_size) we'll use
        size_t max_batch = *std::max_element(prefill_batch_sizes_.begin(), prefill_batch_sizes_.end());
        size_t block_per_req = nblocks / max_batch;
        block_tables_holder_ = infinicore::Tensor::empty(
            {nblocks}, infinicore::DataType::I32, infinicore::context::getDevice());
        set_zeros(block_tables_holder_);

        for (size_t b : prefill_batch_sizes_) {
            for (size_t cs : chunk_sizes_) {
                size_t total_tokens = b * cs;
                if (total_tokens > MAX_TOTAL_TOKENS) {
                    continue;
                }

                size_t bpr = nblocks / b; // block_per_req for this batch size

                InfinilmModel::Input input;

                // input_ids: [1, total_tokens] — all tokens for this batch packed together
                input.input_ids = infinicore::Tensor::empty(
                    {1, total_tokens}, infinicore::DataType::I64, infinicore::context::getDevice());
                set_zeros(input.input_ids.value());

                // position_ids: [total_tokens]
                input.position_ids = infinicore::Tensor::empty(
                    {total_tokens}, infinicore::DataType::I64, infinicore::context::getDevice());
                set_zeros(input.position_ids.value());

                // total_sequence_lengths: [b], set to cs (first-chunk scenario)
                input.total_sequence_lengths = infinicore::Tensor::empty(
                    {b}, infinicore::DataType::I32, infinicore::context::getDevice());
                {
                    std::vector<int32_t> tsl(b, static_cast<int32_t>(cs));
                    infinicore::context::memcpyH2D(
                        input.total_sequence_lengths.value()->data(),
                        tsl.data(), b * sizeof(int32_t), false);
                }

                // input_offsets: [b+1], stride = cs
                input.input_offsets = infinicore::Tensor::empty(
                    {b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
                {
                    std::vector<int32_t> offsets(b + 1);
                    for (size_t i = 0; i <= b; i++) {
                        offsets[i] = static_cast<int32_t>(i * cs);
                    }
                    infinicore::context::memcpyH2D(
                        input.input_offsets.value()->data(),
                        offsets.data(), (b + 1) * sizeof(int32_t), false);
                }

                // cu_seqlens: [b+1], same layout as input_offsets for prefill
                input.cu_seqlens = infinicore::Tensor::empty(
                    {b + 1}, infinicore::DataType::I32, infinicore::context::getDevice());
                {
                    std::vector<int32_t> cu(b + 1);
                    for (size_t i = 0; i <= b; i++) {
                        cu[i] = static_cast<int32_t>(i * cs);
                    }
                    infinicore::context::memcpyH2D(
                        input.cu_seqlens.value()->data(),
                        cu.data(), (b + 1) * sizeof(int32_t), false);
                }

                // block_tables: view into the shared holder [b, bpr]
                input.block_tables = block_tables_holder_->as_strided(
                    {b, bpr}, {(ptrdiff_t)bpr, 1});

                // slot_mapping: [total_tokens]
                input.slot_mapping = infinicore::Tensor::empty(
                    {total_tokens}, infinicore::DataType::I64, infinicore::context::getDevice());
                set_zeros(input.slot_mapping.value());

                barrier_->wait();
                infinicore::context::startGraphRecording();
                auto output = model_->forward(input);
                auto graph = infinicore::context::stopGraphRecording();
                barrier_->wait();

                auto shared_output = std::shared_ptr<InfinilmModel::Output>(
                    new InfinilmModel::Output{infinicore::graph::GraphTensor(output.logits)});

                compiled_map_prefill_[std::make_tuple(b, cs)] =
                    CompiledResult{std::move(input), std::make_tuple(graph, shared_output)};
            }
        }
    }
}

ChunkPrefillCompiler::Compiled ChunkPrefillCompiler::get_compiled(const InfinilmModel::Input &input) {
    if (model_->get_cache_config() == nullptr ||
        !dynamic_cast<const cache::PagedKVCacheConfig *>(model_->get_cache_config())) {
        return {nullptr, nullptr};
    }

    if (!input.block_tables.has_value() || !input.input_ids.has_value()) {
        return {nullptr, nullptr};
    }

    size_t batch_size = input.block_tables.value()->size(0);
    size_t block_per_req = input.block_tables.value()->size(1);
    size_t total_tokens = input.input_ids.value()->size(1);

    // Prefill: total_tokens is a multiple of batch_size, and chunk_size > 1
    if (total_tokens == 0 || total_tokens % batch_size != 0) {
        return {nullptr, nullptr};
    }
    size_t chunk_size = total_tokens / batch_size;
    if (chunk_size <= 1) {
        // Single-token case belongs to decode
        return {nullptr, nullptr};
    }

    auto result = compiled_map_prefill_.find(std::make_tuple(batch_size, chunk_size));
    if (result == compiled_map_prefill_.end()) {
        return {nullptr, nullptr};
    }

    auto &graph_input = result->second.input;

    graph_input.input_ids.value()->copy_from(input.input_ids.value());
    graph_input.position_ids.value()->copy_from(input.position_ids.value());
    graph_input.total_sequence_lengths.value()->copy_from(input.total_sequence_lengths.value());
    graph_input.input_offsets.value()->copy_from(input.input_offsets.value());
    graph_input.cu_seqlens.value()->copy_from(input.cu_seqlens.value());
    graph_input.block_tables.value()->narrow({{1, 0, block_per_req}})->copy_from(input.block_tables.value());
    graph_input.slot_mapping.value()->copy_from(input.slot_mapping.value());

    auto graph = std::get<0>(result->second.compiled);
    auto shared_output = std::shared_ptr<InfinilmModel::Output>(
        new InfinilmModel::Output{std::get<1>(result->second.compiled)->logits->resume_from_blob_()});

    return std::make_tuple(graph, shared_output);
}

} // namespace infinilm::engine
