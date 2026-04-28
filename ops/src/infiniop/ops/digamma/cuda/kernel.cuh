#pragma once
#include <cmath>
#include <type_traits>

namespace op::cuda {

// Digamma for x > 0 using recurrence + asymptotic series.
template <typename T>
__device__ __forceinline__ T digamma_impl(T x) {
    if (x == static_cast<T>(0)) {
        return static_cast<T>(-INFINITY);
    }
    if (x < static_cast<T>(0)) {
        return static_cast<T>(NAN);
    }

    T result = static_cast<T>(0);
    while (x < static_cast<T>(8)) {
        result -= static_cast<T>(1) / x;
        x += static_cast<T>(1);
    }

    const T inv = static_cast<T>(1) / x;
    const T inv2 = inv * inv;

    const T series = inv2 * (static_cast<T>(-1.0 / 12.0) + inv2 * (static_cast<T>(1.0 / 120.0) + inv2 * (static_cast<T>(-1.0 / 252.0) + inv2 * (static_cast<T>(1.0 / 240.0) + inv2 * (static_cast<T>(-1.0 / 132.0))))));

    result += log(x) - static_cast<T>(0.5) * inv + series;
    return result;
}

typedef struct DigammaOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            return __float2half(digamma_impl(xf));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            float xf = __bfloat162float(x);
            return __float2bfloat16_rn(digamma_impl(xf));
        } else if constexpr (std::is_same_v<T, float>) {
            return digamma_impl(x);
        } else { // double
            return digamma_impl(static_cast<double>(x));
        }
    }
} DigammaOp;

} // namespace op::cuda
