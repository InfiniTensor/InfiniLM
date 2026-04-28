#ifndef __GELU_CUDA_H__
#define __GELU_CUDA_H__

#include <cmath>

namespace op::gelu::cuda {

typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {

        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float x_f = __bfloat162float(x);
            float result = 0.5 * x_f * (1 + erf(x_f / sqrt(2.0f)));

            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float x_f = __half2float(x);
            float result = 0.5 * x_f * (1 + erf(x_f / sqrt(2.0f)));

            return __float2half(result);
        } else if constexpr (std::is_same_v<T, float>) {

            return 0.5 * x * (1 + erf(x / sqrt(2.0f)));
        } else {
            return 0.5 * x * (1 + erf(x / sqrt(2.0)));
        }
    }
} GeluOp;

} // namespace op::gelu::cuda

#endif // __GELU_CUDA_H__
