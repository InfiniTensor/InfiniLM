#ifndef __SELU_MOORE_KERNEL_H__
#define __SELU_MOORE_KERNEL_H__

#include <cmath>
#include <type_traits>

namespace op::selu::moore {

constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

typedef struct SeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            float r0 = x0 > 0.0f ? SELU_SCALE * x0 : SELU_SCALE * SELU_ALPHA * (expf(x0) - 1.0f);
            float r1 = x1 > 0.0f ? SELU_SCALE * x1 : SELU_SCALE * SELU_ALPHA * (expf(x1) - 1.0f);
            return __floats2half2_rn(r0, r1);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float result = xf > 0.0f ? SELU_SCALE * xf : SELU_SCALE * SELU_ALPHA * (expf(xf) - 1.0f);
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            float result = xf > 0.0f ? SELU_SCALE * xf : SELU_SCALE * SELU_ALPHA * (expf(xf) - 1.0f);
            return __float2bfloat16_rn(result);
        } else if constexpr (std::is_same_v<T, float>) {
            return x > 0.0f ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * (expf(x) - 1.0f);
        } else { // double
            return x > 0.0 ? static_cast<double>(SELU_SCALE) * x : static_cast<double>(SELU_SCALE) * static_cast<double>(SELU_ALPHA) * (exp(x) - 1.0);
        }
    }
} SeluOp;

} // namespace op::selu::moore

#endif // __SELU_MOORE_KERNEL_H__
