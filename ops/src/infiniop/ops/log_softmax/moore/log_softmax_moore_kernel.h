#ifndef __LOG_SOFTMAX_MOORE_H__
#define __LOG_SOFTMAX_MOORE_H__

#include <cmath>
#include <cstdint>
#include <limits>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::log_softmax::moore {
template <typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}
template <typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    // 32-thread warp reduction
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_max(T val) {
    static __shared__ float shared[32]; // Max 32 warps per block
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -INFINITY;

    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

template <typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}
template <typename T>
__global__ void log_softmax_kernel(
    T *__restrict__ output,      // [Outer, Dim, Inner]
    const T *__restrict__ input, // [Outer, Dim, Inner]
    size_t dim_size,
    size_t inner_size) {
    // 共享内存用于存储 Block Reduction 的结果广播
    __shared__ float s_max;
    __shared__ float s_sum;

    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;

    // 1. 计算当前 Slice 的基地址
    // GridDim.x = Outer * Inner
    size_t outer_idx = bid / inner_size;
    size_t inner_idx = bid % inner_size;
    size_t base_offset = outer_idx * dim_size * inner_size + inner_idx;
    size_t stride = inner_size; // 元素在 Dim 维度的跨度
    float local_max = -INFINITY;
    for (size_t i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float(input[base_offset + i * stride]);
        if (val > local_max) {
            local_max = val;
        }
    }

    // Block Reduction 得到全局 Max
    float global_max = block_reduce_max(local_max);
    // 线程 0 将结果写入共享内存
    if (tid == 0) {
        s_max = global_max;
    }
    __syncthreads();
    // 广播到所有线程
    global_max = s_max;

    // ============================================================
    // Pass 2: Calculate Sum of Exponentials
    // sum(exp(x - max))
    // ============================================================
    float local_sum = 0.0f;
    for (size_t i = tid; i < dim_size; i += blockDim.x) {
        float val = to_float(input[base_offset + i * stride]);
        local_sum += expf(val - global_max);
    }

    // Block Reduction 得到全局 Sum
    float global_sum = block_reduce_sum(local_sum);
    if (tid == 0) {
        s_sum = global_sum;
    }
    __syncthreads();
    global_sum = s_sum; // 广播
    float log_sum_exp = logf(global_sum) + global_max;
    for (size_t i = tid; i < dim_size; i += blockDim.x) {
        size_t idx = base_offset + i * stride;
        float val = to_float(input[idx]);
        // 最终写回
        output[idx] = static_cast<T>(val - log_sum_exp);
    }
}

} // namespace op::log_softmax::moore

#endif // __LOG_SOFTMAX_MOORE_H__
