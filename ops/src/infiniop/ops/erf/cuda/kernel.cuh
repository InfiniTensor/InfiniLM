#pragma once
#include <cmath>
#include <type_traits>

namespace op::cuda {

struct ErfOp {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(T x) const {
        if constexpr (std::is_same_v<T, float>) {
            return erff(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::erf(x);
        } else {
            // For F16/BF16: promote to float, compute, then cast back
            float xf;
            if constexpr (std::is_same_v<T, half>) {
                xf = __half2float(x);
                return __float2half_rn(erff(xf));
            } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
                xf = __bfloat162float(x);
                return __float2bfloat16_rn(erff(xf));
            } else {
                xf = static_cast<float>(x);
                return static_cast<T>(erff(xf));
            }
        }
    }
};

} // namespace op::cuda
