#ifndef __FLIPUD_CUDA_CUH__
#define __FLIPUD_CUDA_CUH__

#include <cmath>

namespace op::flipud::cuda {

constexpr int MAX_DIMS = 8;

template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

struct TensorLayout {
    int ndim;
    size_t shape[MAX_DIMS];
    size_t in_strides[MAX_DIMS];
    size_t out_strides[MAX_DIMS];
};

__device__ __forceinline__ void index_to_coords(size_t index, const TensorLayout &layout, size_t *coords) {
    size_t temp = index;
#pragma unroll
    for (int i = layout.ndim - 1; i >= 0; --i) {
        coords[i] = temp % layout.shape[i];
        temp /= layout.shape[i];
    }
}

__device__ __forceinline__ size_t coords_to_offset(const size_t *coords, const size_t *strides, int ndim) {
    size_t offset = 0;
#pragma unroll
    for (int i = 0; i < ndim; ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

template <typename T>
__global__ void flipud_kernel(
    T *__restrict__ output,
    const T *__restrict__ input,
    size_t numel,
    TensorLayout layout) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        size_t coords[MAX_DIMS];
        index_to_coords(idx, layout, coords);

        size_t out_offset = coords_to_offset(coords, layout.out_strides, layout.ndim);

        coords[0] = layout.shape[0] - 1 - coords[0];

        size_t in_offset = coords_to_offset(coords, layout.in_strides, layout.ndim);

        output[out_offset] = input[in_offset];
    }
}

template <typename T, int PackSize>
__global__ void flipud_kernel_vectorized(
    T *__restrict__ output,
    const T *__restrict__ input,
    size_t num_packs,
    TensorLayout layout) {

    using PackType = Pack<T, PackSize>;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_packs) {
        size_t scalar_idx = idx * PackSize;
        size_t coords[MAX_DIMS];

        index_to_coords(scalar_idx, layout, coords);

        size_t out_offset = coords_to_offset(coords, layout.out_strides, layout.ndim);

        coords[0] = layout.shape[0] - 1 - coords[0];

        size_t in_offset = coords_to_offset(coords, layout.in_strides, layout.ndim);

        *reinterpret_cast<PackType *>(output + out_offset) = *reinterpret_cast<const PackType *>(input + in_offset);
    }
}

} // namespace op::flipud::cuda

#endif // __FLIPUD_CUDA_CUH__
