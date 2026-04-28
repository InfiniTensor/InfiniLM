#ifndef __BROADCAST_TO_CUDA_CUH__
#define __BROADCAST_TO_CUDA_CUH__

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace op::broadcast_to::cuda {

// 最大维度定义，需与 BroadcastToInfo 中的保持一致
static constexpr int MAX_DIM = 8;
struct BroadcastStrides {
    int64_t out_strides[MAX_DIM];
    int64_t in_strides[MAX_DIM];
};
template <typename T>
__global__ void broadcast_kernel(
    T *__restrict__ output,      // Output data pointer
    const T *__restrict__ input, // Input data pointer
    int ndim,
    size_t count,               // Total elements in output
    BroadcastStrides strides) { // Strides passed by value

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        size_t temp_idx = idx;
        size_t input_offset = 0;

// 坐标变换与偏移计算
#pragma unroll
        for (int i = 0; i < MAX_DIM; ++i) {
            if (i >= ndim) {
                break;
            }

            int64_t out_s = strides.out_strides[i];
            int64_t in_s = strides.in_strides[i];
            size_t coord = temp_idx / out_s;
            temp_idx %= out_s;
            input_offset += coord * in_s;
        }

        output[idx] = input[input_offset];
    }
}

} // namespace op::broadcast_to::cuda

#endif // __BROADCAST_TO_CUDA_CUH__
