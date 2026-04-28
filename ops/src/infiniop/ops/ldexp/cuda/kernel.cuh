#ifndef __LDEXP_CUDA_CUH__
#define __LDEXP_CUDA_CUH__

#include <cmath>
#include <cstdint>

namespace op::ldexp::cuda {

static constexpr int MAX_DIMS = 8;

template <typename T>
__device__ __forceinline__ float to_float(T val) {
    return static_cast<float>(val);
}
// 特化 half/bf16 的转换
#if !defined(ENABLE_METAX_API)
template <>
__device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template <>
__device__ __forceinline__ float to_float<cuda_bfloat16>(cuda_bfloat16 val) { return __bfloat162float(val); }
#endif

// ldexp wrapper
template <typename T>
__device__ __forceinline__ T ldexp_wrapper(float x_f, int exp_i) {
    return static_cast<T>(::ldexpf(x_f, exp_i));
}
template <>
__device__ __forceinline__ double ldexp_wrapper<double>(float x_f, int exp_i) { return ::ldexp((double)x_f, exp_i); }

struct KernelShapeInfo {
    int ndim;
    int shape[MAX_DIMS];
    int stride_x[MAX_DIMS];
    int stride_exp[MAX_DIMS];
};

// [修复] 增加 TExp 模板参数
template <typename T, typename TExp>
__global__ void ldexp_broadcast_kernel(
    T *__restrict__ output,
    const T *__restrict__ x,
    const TExp *__restrict__ exp, // 正确的 exp 指针类型
    size_t n,
    KernelShapeInfo info) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = idx; i < n; i += stride) {
        size_t temp_idx = i;
        size_t offset_x = 0;
        size_t offset_exp = 0;

#pragma unroll
        for (int d = info.ndim - 1; d >= 0; --d) {
            int dim_size = info.shape[d];
            int coord = temp_idx % dim_size;
            temp_idx /= dim_size;
            offset_x += coord * info.stride_x[d];
            offset_exp += coord * info.stride_exp[d];
        }

        float x_val = to_float(x[offset_x]);
        // [修复] 将 exp 转换为 float/int
        float exp_val_f = to_float(exp[offset_exp]);

        output[i] = ldexp_wrapper<T>(x_val, static_cast<int>(exp_val_f));
    }
}
} // namespace op::ldexp::cuda
#endif
