#include <cstddef>
#include <type_traits>

namespace op::cuda {

template <typename T>
__global__ void kron_kernel(
    T *output,
    const T *a,
    const T *b,
    size_t total_output,
    size_t ndim,
    const size_t *a_shape,
    const size_t *b_shape,
    const size_t *y_shape,
    const ptrdiff_t *a_strides,
    const ptrdiff_t *b_strides,
    const ptrdiff_t *y_strides) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) {
        return;
    }

    // Convert linear index to coordinates
    size_t temp = idx;
    constexpr size_t kMaxNdim = 8;
    size_t y_coords[kMaxNdim]; // Max 8 dimensions
    for (size_t d = ndim; d-- > 0;) {
        y_coords[d] = temp % y_shape[d];
        temp /= y_shape[d];
    }

    ptrdiff_t a_offset = 0;
    ptrdiff_t b_offset = 0;
    ptrdiff_t y_offset = 0;
    for (size_t d = 0; d < ndim; ++d) {
        const auto y_coord = y_coords[d];
        const auto a_coord = y_coord / b_shape[d];
        const auto b_coord = y_coord % b_shape[d];

        a_offset += static_cast<ptrdiff_t>(a_coord) * a_strides[d];
        b_offset += static_cast<ptrdiff_t>(b_coord) * b_strides[d];
        y_offset += static_cast<ptrdiff_t>(y_coord) * y_strides[d];
    }

    output[y_offset] = a[a_offset] * b[b_offset];
}

} // namespace op::cuda
