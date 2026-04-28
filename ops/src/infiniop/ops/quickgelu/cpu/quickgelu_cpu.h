#ifndef __QUICKGELU_CPU_H__
#define __QUICKGELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

#include <cmath>

ELEMENTWISE_DESCRIPTOR(quickgelu, cpu)

namespace op::quickgelu::cpu {
typedef struct QuickGeluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // quickgelu(x) = x * sigmoid(1.702 * x)
        constexpr T alpha = static_cast<T>(1.702);
        T ax = alpha * x;
        return x / (static_cast<T>(1) + std::exp(-ax));
    }
} QuickGeluOp;
} // namespace op::quickgelu::cpu

#endif // __QUICKGELU_CPU_H__
