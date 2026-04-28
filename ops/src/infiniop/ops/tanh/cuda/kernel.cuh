#ifndef __TANH_CUDA_H__
#define __TANH_CUDA_H__

#include <cmath>

namespace op::tanh::cuda {
typedef struct TanhOp {
    static constexpr size_t num_inputs = 1;

    __device__ __forceinline__ float tanh_f32_func(float x) const {
        return tanhf(x);
    }
    template <typename T>
    __device__ __forceinline__ T operator()(const T &input) const {
        if constexpr (std::is_same_v<T, half2>) {
            float2 vf = __half22float2(input);
            float2 vr = make_float2(tanh_f32_func(vf.x), tanh_f32_func(vf.y));
            return __float22half2_rn(vr);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(input);
            float yf = tanh_f32_func(xf);
            return __float2half_rn(yf);
        } else if constexpr (std::is_same_v<T, cuda_bfloat162>) {
            float f0 = __bfloat162float(__low2bfloat16(input));
            float f1 = __bfloat162float(__high2bfloat16(input));
            float r0 = tanh_f32_func(f0);
            float r1 = tanh_f32_func(f1);
            return __floats2bfloat162_rn(r0, r1);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(input);
            float rf = tanh_f32_func(xf);
            return __float2bfloat16_rn(rf);
        } else if constexpr (std::is_same_v<T, float>) {
            return tanh_f32_func(input);
        } else if constexpr (std::is_same_v<T, double>) {
            return std::tanh(input);
        } else {
            return std::tanh(input);
        }
    }
} TanhOp;
} // namespace op::tanh::cuda

#endif // __TANH_CUDA_H__
