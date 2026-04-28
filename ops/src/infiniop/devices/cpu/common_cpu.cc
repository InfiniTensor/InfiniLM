#include "common_cpu.h"

namespace op::common_cpu {

size_t indexToOffset(
    size_t flat_index,
    size_t ndim,
    const size_t *shape,
    const ptrdiff_t *strides) {
    size_t res = 0;
    for (size_t i = ndim; i-- > 0;) {
        res += (flat_index % shape[i]) * strides[i];
        flat_index /= shape[i];
    }
    return res;
}

size_t getPaddedSize(
    size_t ndim,
    size_t *shape,
    const size_t *pads) {
    size_t total_size = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total_size *= shape[i] + (i < 2 ? 0 : 2 * pads[i - 2]);
    }
    return total_size;
}

std::vector<size_t> getPaddedShape(
    size_t ndim,
    const size_t *shape,
    const size_t *pads) {
    std::vector<size_t> padded_shape(ndim);
    memcpy(padded_shape.data(), shape, ndim * sizeof(size_t));
    for (size_t i = 2; i < ndim; ++i) {
        padded_shape[i] += 2 * pads[i - 2];
    }
    return padded_shape;
}

} // namespace op::common_cpu
