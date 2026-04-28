#pragma once
#include <algorithm>
#include <cmath>
#include <type_traits>

namespace op::relu6::cuda {

struct Relu6Op {
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            float result = fminf(fmaxf(xf, 0.0f), 6.0f);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float result = fminf(fmaxf(xf, 0.0f), 6.0f);
            return __float2half(result);
        } else if constexpr (std::is_same_v<T, float>) {
            return fminf(fmaxf(x, 0.0f), 6.0f);
        } else if constexpr (std::is_same_v<T, double>) {
            return fmin(fmax(x, 0.0), 6.0);
        } else {
            float xf = static_cast<float>(x);
            float result = fminf(fmaxf(xf, 0.0f), 6.0f);
            return static_cast<T>(result);
        }
    }
};

} // namespace op::relu6::cuda
