#include "paged_cache.hpp"

#include "runtime/runtime.hpp"

#include <limits>
#include <stdexcept>

namespace infinilm::models::deepseek_v4 {
namespace {

constexpr size_t kFlashMlaAlignment = 576;
constexpr size_t kIndexQuantBlock = 128;

size_t product(size_t x, size_t y) {
    if (y && x > std::numeric_limits<size_t>::max() / y) {
        throw std::overflow_error("DeepSeek V4 cache size overflows");
    }
    return x * y;
}

size_t align(size_t x, size_t multiple) {
    if (x > std::numeric_limits<size_t>::max() - (multiple - 1)) {
        throw std::overflow_error("DeepSeek V4 cache alignment overflows");
    }
    return (x + multiple - 1) / multiple * multiple;
}

runtime::Tensor zeros(runtime::Runtime &rt, size_t rows, size_t width, runtime::DataType type) {
    const auto bytes = product(product(rows, width), runtime::dsize(type));
    auto result = runtime::Tensor::empty(rt, {rows, width}, type);
    rt.setDeviceMemory(result->data(), 0, bytes);
    return result;
}

runtime::Tensor view_as_flashmla(const runtime::Tensor &storage, size_t page_tokens) {
    if (!storage) return {};
    const auto width = static_cast<runtime::Stride>(kV4FlashMlaWidth);
    const auto stride = static_cast<runtime::Stride>(storage->size(1));
    return storage->as_strided({storage->size(0), page_tokens, 1, kV4FlashMlaWidth},
                               {stride, width, width, 1});
}

} // namespace

runtime::Tensor LayerPagedCache::swa_flashmla_view() const {
    return view_as_flashmla(swa, kV4PageSize);
}

runtime::Tensor LayerPagedCache::compressed_flashmla_view() const {
    if (!compression_ratio) return {};
    return view_as_flashmla(compressed, kV4PageSize / compression_ratio);
}

LayerPagedCache allocate_layer_paged_cache(runtime::Runtime &rt, size_t full_blocks,
                                           size_t swa_blocks, size_t ratio, size_t index_head_dim) {
    rt.assertOwnerThread();
    if (rt.isGraphRecording()) {
        throw std::logic_error("DeepSeek V4 cache allocation cannot run during graph recording");
    }
    if (!full_blocks || !swa_blocks || swa_blocks > full_blocks
        || (ratio != 0 && ratio != 4 && ratio != 128)
        || (ratio == 4 && (!index_head_dim || index_head_dim % kIndexQuantBlock))) {
        throw std::invalid_argument("Invalid DeepSeek V4 cache geometry");
    }

    using runtime::DataType;
    LayerPagedCache cache;
    cache.compression_ratio = ratio;
    cache.swa_blocks = swa_blocks;
    cache.full_blocks = full_blocks;
    cache.swa = zeros(rt, swa_blocks, align(kV4PageSize * kV4FlashMlaWidth, kFlashMlaAlignment),
                      DataType::kUInt8);
    if (!ratio) return cache;

    const auto page_tokens = kV4PageSize / ratio;
    cache.compressed = zeros(rt, full_blocks, align(page_tokens * kV4FlashMlaWidth, kFlashMlaAlignment),
                             DataType::kUInt8);
    const size_t ring = ratio == 4 ? 8 : 128;
    const size_t coeff = ratio == 4 ? 2 : 1;
    if (swa_blocks > std::numeric_limits<size_t>::max() / ring - 2) {
        throw std::overflow_error("DeepSeek V4 compressor ring size overflows");
    }
    const auto state_rows = align((swa_blocks + 1) * ring + 1, ratio);
    cache.compressor_state = zeros(rt, state_rows, 2 * coeff * kV4HeadDim, DataType::kFloat32);
    if (ratio == 4) {
        const auto index_page_bytes =
            product(page_tokens, product(index_head_dim / kIndexQuantBlock, kIndexQuantBlock + sizeof(float)));
        cache.indexer = zeros(rt, full_blocks, index_page_bytes, DataType::kUInt8);
        cache.indexer_state = zeros(rt, state_rows, product(2 * coeff, index_head_dim), DataType::kFloat32);
    }
    return cache;
}

} // namespace infinilm::models::deepseek_v4
