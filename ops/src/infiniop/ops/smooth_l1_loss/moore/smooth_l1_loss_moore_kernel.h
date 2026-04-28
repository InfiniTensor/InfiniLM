#ifndef __SMOOTH_L1_LOSS_MOORE_KERNEL_H__
#define __SMOOTH_L1_LOSS_MOORE_KERNEL_H__

#include <cmath>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

namespace op::smooth_l1_loss::moore {

// ==================================================================
// 1. Type Converter & Atomic Add (Keep existing helpers)
// ==================================================================
template <typename T>
union TypeConverter;

template <>
union TypeConverter<half> {
    half val;
    unsigned short bits;
};

template <>
union TypeConverter<__mt_bfloat16> {
    __mt_bfloat16 val;
    unsigned short bits;
};

template <typename T>
__device__ __forceinline__ void atomic_add_func(T *address, T val) {
    atomicAdd(address, val);
}

template <>
__device__ __forceinline__ void atomic_add_func<half>(half *address, half val) {
    unsigned short *address_as_us = reinterpret_cast<unsigned short *>(address);
    unsigned short old = *address_as_us;
    unsigned short assumed;
    do {
        assumed = old;
        TypeConverter<half> old_converter;
        old_converter.bits = assumed;
        float sum_f = __half2float(old_converter.val) + __half2float(val);
        TypeConverter<half> new_converter;
        new_converter.val = __float2half(sum_f);
        old = atomicCAS(address_as_us, assumed, new_converter.bits);
    } while (assumed != old);
}

template <>
__device__ __forceinline__ void atomic_add_func<__mt_bfloat16>(__mt_bfloat16 *address, __mt_bfloat16 val) {
    unsigned short *address_as_us = reinterpret_cast<unsigned short *>(address);
    unsigned short old = *address_as_us;
    unsigned short assumed;
    do {
        assumed = old;
        TypeConverter<__mt_bfloat16> old_converter;
        old_converter.bits = assumed;
        float sum_f = __bfloat162float(old_converter.val) + __bfloat162float(val);
        TypeConverter<__mt_bfloat16> new_converter;
        new_converter.val = __float2bfloat16(sum_f);
        old = atomicCAS(address_as_us, assumed, new_converter.bits);
    } while (assumed != old);
}

// ==================================================================
// 2. Math Functor
// ==================================================================
struct SmoothL1LossMath {
    template <typename T>
    __device__ __forceinline__ float operator()(T x_val, T y_val, float beta) const {
        float x_f, y_f;
        if constexpr (std::is_same_v<T, half>) {
            x_f = __half2float(x_val);
            y_f = __half2float(y_val);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            x_f = __bfloat162float(x_val);
            y_f = __bfloat162float(y_val);
        } else {
            x_f = static_cast<float>(x_val);
            y_f = static_cast<float>(y_val);
        }

        float diff = x_f - y_f;
        float abs_diff = ::fabsf(diff);
        if (abs_diff < beta) {
            return 0.5f * diff * diff / beta;
        } else {
            return abs_diff - 0.5f * beta;
        }
    }
};

// ==================================================================
// 3. Optimized Kernels
// ==================================================================

// ------------------------------------------------------------------
// Kernel A: Elementwise (No Reduction)
// ------------------------------------------------------------------
template <typename T>
__global__ void smooth_l1_loss_elementwise_kernel(
    const size_t numel,
    const float beta,
    const T *input,
    const T *target,
    T *output) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        float loss_f = SmoothL1LossMath()(input[idx], target[idx], beta);

        if constexpr (std::is_same_v<T, half>) {
            output[idx] = __float2half(loss_f);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            output[idx] = __float2bfloat16(loss_f);
        } else {
            output[idx] = static_cast<T>(loss_f);
        }
    }
}

// ------------------------------------------------------------------
// Kernel B: Block Reduction (For Mean/Sum) - High Precision
// ------------------------------------------------------------------
template <typename T>
__global__ void smooth_l1_loss_reduce_kernel(
    const size_t numel,
    const float beta,
    const T *input,
    const T *target,
    T *output) {

    // 1. Thread-Local Accumulation (in Float32)
    float thread_sum = 0.0f;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < numel; i += stride) {
        thread_sum += SmoothL1LossMath()(input[i], target[i], beta);
    }

    // 2. Shared Memory Reduction
    // Declare dynamic shared memory (size determined at launch)
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    sdata[tid] = thread_sum;
    __syncthreads();

    // Standard Tree Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 3. Block Leader writes to Global Memory
    if (tid == 0) {
        float block_sum = sdata[0];
        // Convert to T only ONCE per block
        T val_to_add;
        if constexpr (std::is_same_v<T, half>) {
            val_to_add = __float2half(block_sum);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            val_to_add = __float2bfloat16(block_sum);
        } else {
            val_to_add = static_cast<T>(block_sum);
        }

        atomic_add_func(output, val_to_add);
    }
}

// ------------------------------------------------------------------
// Kernel C: Mean Scaling
// ------------------------------------------------------------------
template <typename T>
__global__ void avg_scaling_kernel(T *output, size_t numel) {
    if (threadIdx.x == 0) {
        float sum_f;
        if constexpr (std::is_same_v<T, half>) {
            sum_f = __half2float(output[0]);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            sum_f = __bfloat162float(output[0]);
        } else {
            sum_f = static_cast<float>(output[0]);
        }

        float mean_f = sum_f / static_cast<float>(numel);

        if constexpr (std::is_same_v<T, half>) {
            output[0] = __float2half(mean_f);
        } else if constexpr (std::is_same_v<T, __mt_bfloat16>) {
            output[0] = __float2bfloat16(mean_f);
        } else {
            output[0] = static_cast<T>(mean_f);
        }
    }
}

} // namespace op::smooth_l1_loss::moore

#endif // __SMOOTH_L1_LOSS_MOORE_KERNEL_H__
