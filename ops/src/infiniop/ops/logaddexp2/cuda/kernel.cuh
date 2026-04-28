#ifndef __LOGADDEXP2_CUDA_H__
#define __LOGADDEXP2_CUDA_H__

#include <cmath>

namespace op::logaddexp2::cuda {

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
        } else if constexpr (std::is_same_v<T, half> || std::is_same_v<T, cuda_bfloat16>) {
            return static_cast<T>(logaddexp2_func(static_cast<float>(a), static_cast<float>(b)));
        } else if constexpr (std::is_same_v<T, float>) {
            return logaddexp2_func(a, b);
        } else {
            return static_cast<T>(logaddexp2_func(static_cast<double>(a), static_cast<double>(b)));
        }
    }
} LogAddExp2Op;

} // namespace op::logaddexp2::cuda

#endif // __LOGADDEXP2_CUDA_H__
