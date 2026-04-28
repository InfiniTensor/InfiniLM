#ifndef __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_MOORE_KERNEL_H__
#define __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_MOORE_KERNEL_H__

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace op::triplet_margin_with_distance_loss::moore {
__device__ __forceinline__ float to_float(float val) { return val; }
__device__ __forceinline__ float to_float(double val) { return static_cast<float>(val); }
__device__ __forceinline__ float to_float(half val) { return __half2float(val); }
__device__ __forceinline__ float to_float(__mt_bfloat16 val) { return __bfloat162float(val); }
template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    for (int offset = 32 / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
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

// ==================================================================
// Kernel: Triplet Margin Loss
// ==================================================================
template <typename T>
__global__ void triplet_margin_loss_kernel(
    T *__restrict__ output,               // [BatchSize] (仅当 Reduction=None 时使用)
    float *__restrict__ reduction_buffer, // [1] FP32 Accumulator (仅当 Reduction!=None 时使用)
    const T *__restrict__ anchor,
    const T *__restrict__ positive,
    const T *__restrict__ negative,
    size_t feature_dim,
    float margin,
    int swap,
    int reduction, // 0: None, 1: Mean, 2: Sum
    size_t batch_size) {
    size_t batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) {
        return;
    }

    size_t tid = threadIdx.x;
    size_t stride = blockDim.x;

    size_t offset_base = batch_idx * feature_dim;

    float sum_sq_ap = 0.0f;
    float sum_sq_an = 0.0f;
    float sum_sq_pn = 0.0f;

    for (size_t i = tid; i < feature_dim; i += stride) {
        size_t idx = offset_base + i;
        float a = to_float(anchor[idx]);
        float p = to_float(positive[idx]);
        float n = to_float(negative[idx]);

        float diff_ap = a - p;
        sum_sq_ap += diff_ap * diff_ap;

        float diff_an = a - n;
        sum_sq_an += diff_an * diff_an;

        if (swap) {
            float diff_pn = p - n;
            sum_sq_pn += diff_pn * diff_pn;
        }
    }

    float dist_sq_ap = block_reduce_sum(sum_sq_ap);
    float dist_sq_an = block_reduce_sum(sum_sq_an);
    float dist_sq_pn = 0.0f;
    if (swap) {
        dist_sq_pn = block_reduce_sum(sum_sq_pn);
    }

    if (tid == 0) {
        float eps = 1e-6f;
        float dist_ap = sqrtf(dist_sq_ap + eps);
        float dist_an = sqrtf(dist_sq_an + eps);

        if (swap) {
            float dist_pn = sqrtf(dist_sq_pn + eps);
            if (dist_pn < dist_an) {
                dist_an = dist_pn;
            }
        }

        float loss = fmaxf(dist_ap - dist_an + margin, 0.0f);

        if (reduction == 0) { // None
            output[batch_idx] = static_cast<T>(loss);
        } else { // Sum or Mean
            atomicAdd(reduction_buffer, loss);
        }
    }
}

template <typename T>
__global__ void cast_and_scale_kernel(T *output, const float *reduction_buffer, size_t batch_size, int reduction) {
    if (threadIdx.x == 0) {
        float val = reduction_buffer[0];
        if (reduction == 1) {
            val /= static_cast<float>(batch_size);
        }

        output[0] = static_cast<T>(val);
    }
}

} // namespace op::triplet_margin_with_distance_loss::moore

#endif // __TRIPLET_MARGIN_WITH_DISTANCE_LOSS_MOORE_KERNEL_H__
