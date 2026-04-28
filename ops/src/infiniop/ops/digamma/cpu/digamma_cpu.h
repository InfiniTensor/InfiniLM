#ifndef __DIGAMMA_CPU_H__
#define __DIGAMMA_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>
#include <limits>

ELEMENTWISE_DESCRIPTOR(digamma, cpu)

namespace op::digamma::cpu {

// Digamma function implementation for x > 0 using recurrence + asymptotic series.
template <typename T>
T digamma_impl(T x) {
    if (x == static_cast<T>(0)) {
        return -std::numeric_limits<T>::infinity();
    }
    if (x < static_cast<T>(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T result = static_cast<T>(0);

    // Recurrence to push x to a region where the asymptotic series is accurate.
    while (x < static_cast<T>(8)) {
        result -= static_cast<T>(1) / x;
        x += static_cast<T>(1);
    }

    const T inv = static_cast<T>(1) / x;
    const T inv2 = inv * inv;

    // Asymptotic series:
    // psi(x) = log(x) - 1/(2x) - 1/(12 x^2) + 1/(120 x^4) - 1/(252 x^6) + 1/(240 x^8) - 1/(132 x^10) + ...
    const T series = inv2 * (static_cast<T>(-1.0 / 12.0) + inv2 * (static_cast<T>(1.0 / 120.0) + inv2 * (static_cast<T>(-1.0 / 252.0) + inv2 * (static_cast<T>(1.0 / 240.0) + inv2 * (static_cast<T>(-1.0 / 132.0))))));

    result += std::log(x) - static_cast<T>(0.5) * inv + series;
    return result;
}

typedef struct DigammaOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return digamma_impl(x);
    }
} DigammaOp;
} // namespace op::digamma::cpu

#endif // __DIGAMMA_CPU_H__
