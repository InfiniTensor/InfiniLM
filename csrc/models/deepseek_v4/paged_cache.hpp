#pragma once

#include "runtime/tensor.hpp"

namespace infinilm::models::deepseek_v4 {

constexpr size_t kV4PageSize = 256;
constexpr size_t kV4FlashMlaWidth = 584;
constexpr size_t kV4HeadDim = 512;

struct LayerPagedCache {
    runtime::Tensor swa;
    runtime::Tensor compressed;
    runtime::Tensor indexer;
    runtime::Tensor compressor_state;
    runtime::Tensor indexer_state;
    size_t compression_ratio{};
    size_t swa_blocks{};
    size_t full_blocks{};

    // FlashMLA expects a 4D byte view, with an aligned stride between pages.
    runtime::Tensor swa_flashmla_view() const;
    runtime::Tensor compressed_flashmla_view() const;
};

// Allocate storage before graph capture. Scheduler-owned full-to-SWA page
// mapping and compressor state updates are separate from this allocation.
LayerPagedCache allocate_layer_paged_cache(
    runtime::Runtime &runtime, size_t full_blocks, size_t swa_blocks,
    size_t compression_ratio, size_t index_head_dim = 128);

} // namespace infinilm::models::deepseek_v4
