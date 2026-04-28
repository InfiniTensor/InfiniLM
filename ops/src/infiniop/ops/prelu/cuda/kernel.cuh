#pragma once
#include <type_traits>

namespace op::prelu::cuda {

typedef struct PreluOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    __device__ __forceinline__ T operator()(const T &x, const T &weight) const {
        if constexpr (std::is_same_v<T, half2>) {
            float x0 = __low2float(x);
            float x1 = __high2float(x);
            float w0 = __low2float(weight);
            float w1 = __high2float(weight);
            float r0 = x0 > 0.0f ? x0 : w0 * x0;
            float r1 = x1 > 0.0f ? x1 : w1 * x1;
            return __floats2half2_rn(r0, r1);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float wf = __half2float(weight);
            float result = xf > 0.0f ? xf : wf * xf;
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            float wf = __bfloat162float(weight);
            float result = xf > 0.0f ? xf : wf * xf;
            return __float2bfloat16_rn(result);
        } else {
            return x > 0 ? x : weight * x;
        }
    }
} PreluOp;

} // namespace op::prelu::cuda
