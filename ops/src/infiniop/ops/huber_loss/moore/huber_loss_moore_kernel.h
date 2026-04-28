#ifndef __HUBER_LOSS_MOORE_KERNEL_H__
#define __HUBER_LOSS_MOORE_KERNEL_H__

#include <cmath>
#include <cstdio>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

namespace op::huber_loss::moore {
template <typename T>
__device__ __forceinline__ float to_float(T val) {
    if constexpr (std::is_same_v<T, half>) {
        return __half2float(val);
    } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
        return __bfloat162float(val);
    } else {
        return static_cast<float>(val);
    }
}

template <typename T>
__device__ __forceinline__ T from_float(float val) {
    if constexpr (std::is_same_v<T, half>) {
        return __float2half(val);
    } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
        return __float2bfloat16(val);
    } else {
        return static_cast<T>(val);
    }
}
__device__ __forceinline__ float warpReduceSum(float val) {
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(mask, val, offset);
    }
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // Max 1024 threads / 32 warps
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

struct HuberLossFunctor {
    float delta;
    float half_delta; // 预计算 0.5 * delta

    __host__ __device__ HuberLossFunctor(float delta_val)
        : delta(delta_val), half_delta(0.5f * delta_val) {}

    // Huber Loss 计算:
    // if |x - y| < delta: 0.5 * (x - y)^2
    // else: delta * (|x - y| - 0.5 * delta)
    __device__ __forceinline__ float compute(float input_val, float target_val) const {
        float diff = input_val - target_val;
        float abs_diff = std::abs(diff);

        if (abs_diff < delta) {
            return 0.5f * diff * diff;
        } else {
            return delta * (abs_diff - half_delta);
        }
    }
};

// ==================================================================
// Kernel 1: Reduction = None (Element-wise output)
// ==================================================================
template <typename T>
__global__ void huber_loss_kernel(
    T *__restrict__ output,       // [N]
    const T *__restrict__ input,  // [N]
    const T *__restrict__ target, // [N]
    size_t count,                 // Total elements (numel)
    HuberLossFunctor functor) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < count) {
        float in_val = to_float(input[idx]);
        float tg_val = to_float(target[idx]);

        float loss = functor.compute(in_val, tg_val);

        output[idx] = from_float<T>(loss);
    }
}

// ==================================================================
// Kernel 2: Reduction = Mean / Sum (Scalar output)
// ==================================================================
template <typename T>
__global__ void huber_loss_reduce_kernel(
    float *output,                // [1] Accumulator (Float)
    const T *__restrict__ input,  // [N]
    const T *__restrict__ target, // [N]
    size_t count,                 // Total elements
    HuberLossFunctor functor,
    float scale // Mean模式传 1/N, Sum模式传 1.0
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    float local_sum = 0.0f;

    // Grid-Stride Loop over all elements
    for (size_t i = idx; i < count; i += stride) {
        float in_val = to_float(input[i]);
        float tg_val = to_float(target[i]);

        local_sum += functor.compute(in_val, tg_val);
    }

    // Block Reduction
    float block_sum = blockReduceSum(local_sum);

    // Global Atomic Add (Reduce to scalar)
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum * scale);
    }
}

// ==================================================================
// Helper: Cast float result to T (used for scalar output)
// ==================================================================
template <typename T>
__global__ void cast_float_to_t(T *output, const float *src) {
    *output = from_float<T>(*src);
}

} // namespace op::huber_loss::moore

#endif // __HUBER_LOSS_MOORE_KERNEL_H__
