#ifndef __FLOAT_POWER_MOORE_KERNEL_H__
#define __FLOAT_POWER_MOORE_KERNEL_H__

#include <cmath>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#include <type_traits>

namespace op::float_power::moore {

// ==================================================================
// 类型转换辅助函数 (适配 MUSA)
// ==================================================================
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

// ==================================================================
// 基础定义: 向量化数据打包结构
// ==================================================================
template <typename T, int N>
struct alignas(sizeof(T) * N) Pack {
    T val[N];
};

// ==================================================================
// Functor: 仅负责核心数学计算逻辑
// ==================================================================
struct FloatPowerFunctor {
    template <typename T_IN>
    __device__ __forceinline__ float compute(const T_IN &input, float exponent_val) const {
        // 使用 to_float 辅助函数处理 FP16/BF16
        float in_f = to_float(input);
        return powf(in_f, exponent_val);
    }
};

// ==================================================================
// 1. 通用处理 Kernel (Grid-Stride Loop)
// ==================================================================
template <typename T_OUT, typename T_IN, typename T_EXP>
__global__ void float_power_kernel(
    T_OUT *__restrict__ output,
    const T_IN *__restrict__ input,
    const T_EXP *__restrict__ exponent,
    float scalar_exponent,
    bool is_scalar,
    size_t numel,
    FloatPowerFunctor functor) {

    // Grid-Stride Loop
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < numel;
         idx += blockDim.x * gridDim.x) {

        float exp_val_f = is_scalar ? scalar_exponent : to_float(exponent[idx]);
        output[idx] = from_float<T_OUT>(functor.compute(input[idx], exp_val_f));
    }
}

// ==================================================================
// 2. 标量模式向量化 Kernel
// ==================================================================
template <typename T_OUT, typename T_IN, int PackSize>
__global__ void float_power_kernel_vectorized_scalar(
    T_OUT *__restrict__ output,
    const T_IN *__restrict__ input,
    float scalar_exponent,
    size_t num_packs,
    FloatPowerFunctor functor) {

    using PackTypeIn = Pack<T_IN, PackSize>;
    using PackTypeOut = Pack<T_OUT, PackSize>;

    auto in_vec = reinterpret_cast<const PackTypeIn *>(input);
    auto out_vec = reinterpret_cast<PackTypeOut *>(output);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_packs) {
        PackTypeIn in_pack = in_vec[idx];
        PackTypeOut out_pack;

#pragma unroll
        for (int i = 0; i < PackSize; ++i) {
            out_pack.val[i] = from_float<T_OUT>(functor.compute(in_pack.val[i], scalar_exponent));
        }
        out_vec[idx] = out_pack;
    }
}

// ==================================================================
// 3. 张量模式向量化 Kernel
// ==================================================================
template <typename T_OUT, typename T_IN, int PackSize>
__global__ void float_power_kernel_vectorized_tensor(
    T_OUT *__restrict__ output,
    const T_IN *__restrict__ input,
    const T_IN *__restrict__ exponent,
    size_t num_packs,
    FloatPowerFunctor functor) {

    using PackTypeIn = Pack<T_IN, PackSize>;
    using PackTypeOut = Pack<T_OUT, PackSize>;

    auto in_vec = reinterpret_cast<const PackTypeIn *>(input);
    auto exp_vec = reinterpret_cast<const PackTypeIn *>(exponent);
    auto out_vec = reinterpret_cast<PackTypeOut *>(output);

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_packs) {
        PackTypeIn in_pack = in_vec[idx];
        PackTypeIn exp_pack = exp_vec[idx];
        PackTypeOut out_pack;

#pragma unroll
        for (int i = 0; i < PackSize; ++i) {
            float e = to_float(exp_pack.val[i]);
            out_pack.val[i] = from_float<T_OUT>(functor.compute(in_pack.val[i], e));
        }
        out_vec[idx] = out_pack;
    }
}

} // namespace op::float_power::moore

#endif // __FLOAT_POWER_MOORE_KERNEL_H__
