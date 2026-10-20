#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace infinilm::runtime::detail {
struct CpuCopyStatsR2515 {
    std::uint64_t blocks = 0, block_bytes = 0, scalar_leaves = 0, scalar_bytes = 0;
};

// Caller preserves the original gather-then-scatter rule for overlapping views.
// Coalesce only a suffix contiguous in BOTH views; every other layout retains
// the original logical row-major recursion, including broadcasts and reversals.
template<class Shape, class Strides>
void copy_cpu_strided_r2515(std::byte *dst, const std::byte *src,
                           const Shape &shape, const Strides &ds, const Strides &ss,
                           std::size_t element_size, std::size_t dim,
                           CpuCopyStatsR2515 &stats) {
    if (dim == shape.size()) {
        std::memmove(dst, src, element_size);
        ++stats.scalar_leaves; stats.scalar_bytes += element_size;
        return;
    }
    std::size_t elements = 1;
    bool contiguous = true;
    for (std::size_t j = shape.size(); j > dim; --j) {
        const auto axis = j - 1;
        if (shape[axis] == 0) return;
        if (shape[axis] > 1) {
            if (elements > std::size_t(std::numeric_limits<std::ptrdiff_t>::max())
                || ds[axis] != static_cast<std::ptrdiff_t>(elements)
                || ss[axis] != static_cast<std::ptrdiff_t>(elements)) contiguous = false;
        }
        if (shape[axis] > std::numeric_limits<std::size_t>::max() / elements) {
            contiguous = false;
            break;
        }
        elements *= shape[axis];
    }
    if (contiguous && element_size != 0
        && elements <= std::numeric_limits<std::size_t>::max() / element_size) {
        const auto bytes = elements * element_size;
        std::memmove(dst, src, bytes);
        ++stats.blocks; stats.block_bytes += bytes;
        return;
    }
    const auto bytes = static_cast<std::ptrdiff_t>(element_size);
    for (std::size_t i = 0; i < shape[dim]; ++i) {
        copy_cpu_strided_r2515(dst + std::ptrdiff_t(i) * ds[dim] * bytes,
                              src + std::ptrdiff_t(i) * ss[dim] * bytes,
                              shape, ds, ss, element_size, dim + 1, stats);
    }
}
} // namespace infinilm::runtime::detail
