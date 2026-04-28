#ifndef __LOGADDEXP2_MOORE_KERNEL_H__
#define __LOGADDEXP2_MOORE_KERNEL_H__

#include <cmath>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::logaddexp2::moore {

// ==================================================================
// 1. Math Helpers
// ==================================================================
__device__ __forceinline__ float logaddexp2_func(float a, float b) {
    float max_val = fmaxf(a, b);
    float min_val = fminf(a, b);
    return max_val + log2f(1.0f + exp2f(min_val - max_val));
}

__device__ __forceinline__ double logaddexp2_func(double a, double b) {
    double max_val = fmax(a, b);
    double min_val = fmin(a, b);
    return max_val + log2(1.0 + exp2(min_val - max_val));
}

// ==================================================================
// 2. Functor Definition
// ==================================================================
typedef struct LogAddExp2Op {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 fa = __half22float2(a);
            float2 fb = __half22float2(b);
            float2 res;
            res.x = logaddexp2_func(fa.x, fb.x);
            res.y = logaddexp2_func(fa.y, fb.y);
            return __float22half2_rn(res);
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, __mt_bfloat16>) {
            return static_cast<T>(logaddexp2_func(static_cast<float>(a), static_cast<float>(b)));
        } else if constexpr (std::is_same_v<T, float>) {
            return logaddexp2_func(a, b);
        } else {
            return static_cast<T>(logaddexp2_func(static_cast<double>(a), static_cast<double>(b)));
        }
    }
} LogAddExp2Op;

// ==================================================================
// 3. Kernel Definition
// ==================================================================
template <typename T>
__global__ void logaddexp2_kernel(
    T *output,
    const T *a,
    const T *b,
    size_t n) {

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    LogAddExp2Op op;

    for (size_t i = idx; i < n; i += stride) {
        output[i] = op(a[i], b[i]);
    }
}

} // namespace op::logaddexp2::moore

#endif // __LOGADDEXP2_MOORE_KERNEL_H__
