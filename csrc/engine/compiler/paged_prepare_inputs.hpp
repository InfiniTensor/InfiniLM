#pragma once

#include "runtime/runtime.hpp"
#include "runtime/tensor.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace infinilm::engine {

// One decode token per request, each in a different physical block. Never let
// concurrent warmup requests write the same slot. Unused table entries are -1.
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
        || slots->size(0) != blocks->size(0)
        || slots->size(0) > num_blocks || num_blocks == 0 || block_size == 0
        || num_blocks > static_cast<size_t>(std::numeric_limits<int32_t>::max())
        || block_size > static_cast<size_t>(std::numeric_limits<int64_t>::max()) / num_blocks) {
        throw std::invalid_argument("paged prepare requires disjoint valid block/slot buffers");
    }
    const auto batch = slots->size(0);
    std::vector<int32_t> table(blocks->numel(), -1);
    std::vector<int64_t> positions(batch);
    for (size_t row = 0; row < batch; ++row) {
        table[row * blocks->size(1)] = static_cast<int32_t>(row);
        positions[row] = static_cast<int64_t>(row * block_size);
    }
    runtime.memcpyH2D(blocks->data(), table.data(), table.size() * sizeof(int32_t), false);
    runtime.memcpyH2D(slots->data(), positions.data(), positions.size() * sizeof(int64_t), false);
}

} // namespace infinilm::engine
