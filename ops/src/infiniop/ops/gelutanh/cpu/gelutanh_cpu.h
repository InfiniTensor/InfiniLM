#ifndef __GELUTANH_CPU_H__
#define __GELUTANH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

#include <cmath>

ELEMENTWISE_DESCRIPTOR(gelutanh, cpu)

namespace op::gelutanh::cpu {
typedef struct GeluTanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        constexpr T alpha = static_cast<T>(0.7978845608); // sqrt(2/pi)
        constexpr T beta = static_cast<T>(0.044715);
        T inner = alpha * (x + beta * x * x * x);
        return x * static_cast<T>(0.5) * (static_cast<T>(1) + std::tanh(inner));
    }
} GeluTanhOp;
} // namespace op::gelutanh::cpu

#endif // __GELUTANH_CPU_H__
