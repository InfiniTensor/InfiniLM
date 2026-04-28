#ifndef __GELUTANH_CUDA_H__
#define __GELUTANH_CUDA_H__

#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include <cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace op::gelutanh::cuda {

typedef struct GeluTanhOp {
public:
    static constexpr size_t num_inputs = 1;

    // GELU-Tanh constants
    // static constexpr float alpha = std::sqrt(2.0 / M_PI);
    // static constexpr float beta = 0.044715f;
    static constexpr float alpha = 0.7978845608f; // sqrt(2/pi)
    static constexpr float beta = 0.044715f;
    // f32 tanh helper
    __device__ __forceinline__ float tanh_f32_func(float x) const {
        return tanhf(x);
    }

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // half2 -> float2
            float2 vf = __half22float2(x);
            float inner_x0 = alpha * (vf.x + beta * vf.x * vf.x * vf.x);
            float inner_x1 = alpha * (vf.y + beta * vf.y * vf.y * vf.y);
            float2 vr = make_float2(tanh_f32_func(inner_x0) * 0.5f + 0.5f,
                                    tanh_f32_func(inner_x1) * 0.5f + 0.5f);
            return __hmul2(x, __float22half2_rn(vr)); // y = x * 0.5 * (1 + tanh(...))
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float inner = alpha * (xf + beta * xf * xf * xf);
            float yf = xf * 0.5f * (1.0f + tanh_f32_func(inner));
            return __float2half_rn(yf);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float xf = __bfloat162float(x);
            float inner = alpha * (xf + beta * xf * xf * xf);
            float yf = xf * 0.5f * (1.0f + tanh_f32_func(inner));
            return __float2bfloat16(yf);
        } else if constexpr (std::is_same_v<T, float>) {
            float inner = alpha * (x + beta * x * x * x);
            return x * 0.5f * (1.0f + tanh_f32_func(inner));
        } else { // double
            double inner = alpha * (x + beta * x * x * x);
            return x * 0.5 * (1.0 + std::tanh(inner));
        }
    }

} GeluTanhOp;

} // namespace op::gelutanh::cuda

#endif // __GELUTANH_CUDA_H__
