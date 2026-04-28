#ifndef __LERP_MOORE_H__
#define __LERP_MOORE_H__

#include <cmath>
#include <cstdio>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::lerp::moore {

// ==================================================================
// 辅助函数: 广播坐标映射
// ==================================================================
__device__ __forceinline__ int64_t get_element_offset(
    size_t linear_idx,
    int ndim,
    const int64_t *__restrict__ shape,   // Output Shape
    const int64_t *__restrict__ strides) // Input Effective Strides
{
    int64_t offset = 0;
    size_t remainder = linear_idx;

// 从倒数第 1 维开始向第 0 维反向重构坐标
#pragma unroll
    for (int i = ndim - 1; i >= 0; --i) {
        int64_t dim_size = shape[i];
        int64_t coord = remainder % dim_size;
        remainder /= dim_size;

        // stride 为 0 表示该维度被广播，否则累加物理偏移
        offset += coord * strides[i];
    }
    return offset;
}

// ==================================================================
// Kernel: Lerp
// ==================================================================
template <typename T>
__global__ void lerp_kernel(
    T *__restrict__ output,
    const T *__restrict__ start,
    const T *__restrict__ end,
    const T *__restrict__ weight, // nullptr 表示标量模式
    float weight_scalar,
    size_t numel,
    int ndim,
    const int64_t *__restrict__ shape,         // Output Shape [ndim]
    const int64_t *__restrict__ start_strides, // Broadcasted Strides for Start [ndim]
    const int64_t *__restrict__ end_strides,   // Broadcasted Strides for End [ndim]
    const int64_t *__restrict__ weight_strides // Broadcasted Strides for Weight [ndim] (Optional)
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numel) {
        // 1. 计算 Start 和 End 的偏移量 (支持广播)
        int64_t off_start = get_element_offset(idx, ndim, shape, start_strides);
        int64_t off_end = get_element_offset(idx, ndim, shape, end_strides);

        float s = static_cast<float>(start[off_start]);
        float e = static_cast<float>(end[off_end]);
        float w;

        // 2. 获取权重 (Tensor 或 Scalar)
        if (weight != nullptr) {
            int64_t off_weight = get_element_offset(idx, ndim, shape, weight_strides);
            w = static_cast<float>(weight[off_weight]);
        } else {
            w = weight_scalar;
        }

        // 3. 计算公式: output = start + weight * (end - start)
        float res = s + w * (e - s);

        output[idx] = static_cast<T>(res);
    }
}

} // namespace op::lerp::moore

#endif // __LERP_MOORE_H__
