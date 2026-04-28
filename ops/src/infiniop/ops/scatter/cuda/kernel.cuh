#ifndef __SCATTER_CUDA_CUH__
#define __SCATTER_CUDA_CUH__

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace op::scatter::cuda {

constexpr int MAX_DIMS = 8;

struct TensorGeometry {
    int ndim;
    int64_t updates_shape[MAX_DIMS];
    int64_t updates_strides[MAX_DIMS];
    int64_t output_strides[MAX_DIMS];
    int64_t indices_strides[MAX_DIMS];
};
__device__ __forceinline__ float to_float(float val) { return val; }
__device__ __forceinline__ float to_float(double val) { return static_cast<float>(val); }
__device__ __forceinline__ float to_float(half val) { return __half2float(val); }
__device__ __forceinline__ float to_float(cuda_bfloat16 val) { return __bfloat162float(val); }

template <typename T>
__device__ __forceinline__ T from_float(float val) { return static_cast<T>(val); }
template <>
__device__ __forceinline__ half from_float<half>(float val) { return __float2half(val); }
template <>
__device__ __forceinline__ cuda_bfloat16 from_float<cuda_bfloat16>(float val) { return __float2bfloat16(val); }

__device__ __forceinline__ void offset_to_coords(int64_t offset, int ndim, const int64_t *shape, int64_t *coords) {
#pragma unroll
    for (int i = ndim - 1; i >= 0; --i) {
        coords[i] = offset % shape[i];
        offset /= shape[i];
    }
}

__device__ __forceinline__ int64_t coords_to_offset(int ndim, const int64_t *coords, const int64_t *strides) {
    int64_t offset = 0;
#pragma unroll
    for (int i = 0; i < ndim; ++i) {
        offset += coords[i] * strides[i];
    }
    return offset;
}

template <typename T, typename IdxT>
__global__ void scatter_kernel(
    T *__restrict__ output,
    const T *__restrict__ updates,
    const IdxT *__restrict__ indices,
    TensorGeometry geometry,
    int axis,
    int reduction,
    size_t num_updates) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    int64_t coords[MAX_DIMS];

    for (size_t i = idx; i < num_updates; i += stride) {
        offset_to_coords(static_cast<int64_t>(i), geometry.ndim, geometry.updates_shape, coords);

        int64_t upd_offset = coords_to_offset(geometry.ndim, coords, geometry.updates_strides);
        T upd_val = updates[upd_offset];

        // FIX: 使用 indices_strides 计算 offset
        int64_t idx_offset = coords_to_offset(geometry.ndim, coords, geometry.indices_strides);
        IdxT idx_val = indices[idx_offset];

        coords[axis] = static_cast<int64_t>(idx_val);
        int64_t out_offset = coords_to_offset(geometry.ndim, coords, geometry.output_strides);

        if (reduction == 0) {
            output[out_offset] = upd_val;
        } else if (reduction == 1) {
            float existing = to_float(output[out_offset]);
            float update = to_float(upd_val);
            output[out_offset] = from_float<T>(existing + update);
        } else if (reduction == 2) {
            float existing = to_float(output[out_offset]);
            float update = to_float(upd_val);
            output[out_offset] = from_float<T>(existing * update);
        }
    }
}

} // namespace op::scatter::cuda

#endif // __SCATTER_CUDA_CUH__
