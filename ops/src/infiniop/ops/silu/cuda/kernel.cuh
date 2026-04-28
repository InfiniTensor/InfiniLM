#ifndef __SILU_CUDA_H__
#define __SILU_CUDA_H__

#include <cmath>

namespace op::silu::cuda {

typedef struct SiluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // half2向量化优化
            return __hmul2(x, __h2div(__float2half2_rn(1.0f),
                                      __hadd2(__float2half2_rn(1.0f), h2exp(__hneg2(x)))));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            // BF16
            const float x_f = __bfloat162float(x);
            return __float2bfloat16(x_f / (1.0f + __expf(-x_f)));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16
            const float x_f = __half2float(x);
            return __float2half(x_f / (1.0f + __expf(-x_f)));
        } else if constexpr (std::is_same_v<T, float>) {
            // FP32
            return x * (1.0f / (1.0f + __expf(-x)));
        } else if constexpr (std::is_same_v<T, double>) {
            // FP64
            return x / (1.0 + exp(-x));
        }
    }
} SiluOp;

} // namespace op::silu::cuda

#endif // __SILU_CUDA_H__
