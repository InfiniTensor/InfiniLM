#ifndef __SCATTER_MOORE_KERNEL_H__
#define __SCATTER_MOORE_KERNEL_H__

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace op::scatter::moore {

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
__device__ __forceinline__ float to_float(__mt_bfloat16 val) { return __bfloat162float(val); }

template <typename T>
__device__ __forceinline__ T from_float(float val) { return static_cast<T>(val); }
template <>
__device__ __forceinline__ half from_float<half>(float val) { return __float2half(val); }
template <>
__device__ __forceinline__ __mt_bfloat16 from_float<__mt_bfloat16>(float val) { return __float2bfloat16(val); }

// ==================================================================
// 坐标/偏移计算逻辑 (保持不变)
// ==================================================================

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

// ==================================================================
// Scatter Kernel 实现
// ==================================================================

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
        // 1. 根据 updates 的线性索引反推多维坐标
        offset_to_coords(static_cast<int64_t>(i), geometry.ndim, geometry.updates_shape, coords);

        // 2. 获取 updates 中的值
        int64_t upd_offset = coords_to_offset(geometry.ndim, coords, geometry.updates_strides);
        T upd_val = updates[upd_offset];

        // 3. 获取对应的 indices 值 (使用 indices_strides)
        int64_t idx_offset = coords_to_offset(geometry.ndim, coords, geometry.indices_strides);
        IdxT idx_val = indices[idx_offset];

        // 4. 将坐标中的 axis 维度替换为 index 的值，计算输出偏移
        coords[axis] = static_cast<int64_t>(idx_val);
        int64_t out_offset = coords_to_offset(geometry.ndim, coords, geometry.output_strides);
        if (reduction == 0) { // None
            output[out_offset] = upd_val;
        } else if (reduction == 1) { // Add
            float existing = to_float(output[out_offset]);
            float update = to_float(upd_val);
            output[out_offset] = from_float<T>(existing + update);
        } else if (reduction == 2) { // Multiply
            float existing = to_float(output[out_offset]);
            float update = to_float(upd_val);
            output[out_offset] = from_float<T>(existing * update);
        }
    }
}

} // namespace op::scatter::moore

#endif // __SCATTER_MOORE_KERNEL_H__
