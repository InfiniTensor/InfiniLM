#ifndef __ERFINV_CPU_H__
#define __ERFINV_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>
#include <limits>

ELEMENTWISE_DESCRIPTOR(erfinv, cpu)

namespace op::erfinv::cpu {

// Inverse error function implementation using Newton's method
template <typename T>
T erfinv_impl(T x) {
    // Domain: x in (-1, 1)
    if (x == 1.0) {
        return std::numeric_limits<T>::infinity();
    }
    if (x == -1.0) {
        return -std::numeric_limits<T>::infinity();
    }
    if (x > 1.0 || x < -1.0) {
        return std::numeric_limits<T>::quiet_NaN();
    }
    if (x == 0.0) {
        return 0.0;
    }

    // Use Newton's method to solve erf(y) = x
    T y = x; // Initial guess
    const int max_iter = 10;
    const T tol = static_cast<T>(1e-10);

    for (int i = 0; i < max_iter; ++i) {
        T erf_y = std::erf(y);
        T derf_dy = T(2.0) / T(std::sqrt(3.14159265358979323846) * std::exp(-y * y));
        T error = erf_y - x;
        if (std::abs(error) < tol) {
            break;
        }
        y = y - error / derf_dy;
    }
    return y;
}

typedef struct ErfinvOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return erfinv_impl(x);
    }
} ErfinvOp;
} // namespace op::erfinv::cpu

#endif // __ERFINV_CPU_H__
