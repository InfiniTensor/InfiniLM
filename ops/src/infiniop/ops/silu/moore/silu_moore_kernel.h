#ifndef __SILU_MOORE_KERNEL_H__
#define __SILU_MOORE_KERNEL_H__

#include <cmath>

namespace op::silu::moore {

typedef struct SiluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // half2 vectorized optimization
            return __hmul2(x, __h2div(__float2half2_rn(1.0f),
                                      __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(x)))));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16: convert to float, calculate, then convert back for MUSA platform compatibility
            float x_f = __half2float(x);
            float sigmoid_f = 1.0f / (1.0f + __expf(-x_f));
            return __float2half(x_f * sigmoid_f);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16: convert to float, calculate, then convert back
            float x_f = __bfloat162float(x);
            float sigmoid_f = 1.0f / (1.0f + __expf(-x_f));
            return __float2bfloat16_rn(x_f * sigmoid_f);
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32: use __frcp_rn and __expf for moore platform compatibility
            return __fmul_rn(x, __frcp_rn(__fadd_rn(1.0f, __expf(-x))));
        } else if constexpr (std::is_same_v<T, double>) {
            // FP64
            return x / (1.0 + exp(-x));
        }
    }
} SiluOp;

} // namespace op::silu::moore

#endif // __SILU_MOORE_KERNEL_H__
