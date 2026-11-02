#pragma once

#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infinilm::engine {

// A fixed number of tokens per request, each request in a different physical
// block. Never let concurrent warmup requests write the same slot. Unused table
// entries are -1.
inline void initialize_paged_prepare_metadata(
    infinilm::runtime::Runtime &runtime,
    infinilm::runtime::Tensor blocks,
    infinilm::runtime::Tensor slots,
    size_t num_blocks,
    size_t block_size) {
    runtime.assertOwnerThread();
    if (runtime.isGraphRecording() || !blocks || !slots
        || blocks->device() != runtime.device() || slots->device() != runtime.device()
        || blocks->dtype() != infinilm::runtime::DataType::kInt32
        || slots->dtype() != infinilm::runtime::DataType::kInt64
        || blocks->ndim() != 2 || slots->ndim() != 1
        || !blocks->is_contiguous() || !slots->is_contiguous()
        || blocks->size(0) == 0 || blocks->size(1) == 0
        || slots->size(0) == 0 || slots->size(0) % blocks->size(0) != 0
        || blocks->size(0) > num_blocks || num_blocks == 0 || block_size == 0
        || num_blocks > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        || block_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / num_blocks) {
        throw std::invalid_argument("paged prepare requires disjoint valid block/slot buffers");
    }
    const auto batch = blocks->size(0);
    const auto query_width = slots->size(0) / batch;
    if (query_width > block_size) {
        throw std::invalid_argument("paged prepare query width exceeds block size");
    }
    std::vector<int32_t> table(blocks->numel(), -1);
    std::vector<int64_t> positions(slots->size(0));
    for (size_t row = 0; row < batch; ++row) {
        table[row * blocks->size(1)] = static_cast<int32_t>(row);
        for (size_t token = 0; token < query_width; ++token) {
            positions[row * query_width + token]
                = static_cast<int64_t>(row * block_size + token);
        }
    }
    runtime.memcpyH2D(blocks->data(), table.data(), table.size() * sizeof(int32_t), false);
    runtime.memcpyH2D(slots->data(), positions.data(), positions.size() * sizeof(int64_t), false);
}

} // namespace infinilm::engine
