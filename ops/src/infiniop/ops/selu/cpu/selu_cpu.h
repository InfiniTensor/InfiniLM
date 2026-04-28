#ifndef __SELU_CPU_H__
#define __SELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(selu, cpu)

namespace op::selu::cpu {
// SELU constants: alpha = 1.6732632423543772848170429916717, scale = 1.0507009873554804934193349852946
constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

typedef struct SeluOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, float>) {
            return x > 0.0f ? SELU_SCALE * x : SELU_SCALE * SELU_ALPHA * (std::exp(x) - 1.0f);
        } else if constexpr (std::is_same_v<T, double>) {
            return x > 0.0 ? static_cast<double>(SELU_SCALE) * x : static_cast<double>(SELU_SCALE) * static_cast<double>(SELU_ALPHA) * (std::exp(x) - 1.0);
        } else {
            // For half types, use float computation
            float xf = static_cast<float>(x);
            float result = xf > 0.0f ? SELU_SCALE * xf : SELU_SCALE * SELU_ALPHA * (std::exp(xf) - 1.0f);
            return static_cast<T>(result);
        }
    }
} SeluOp;
} // namespace op::selu::cpu

#endif // __SELU_CPU_H__
