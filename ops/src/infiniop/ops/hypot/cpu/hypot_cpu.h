#ifndef __HYPOT_CPU_H__
#define __HYPOT_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(hypot, cpu)

#include <cmath>
#include <type_traits>

namespace op::hypot::cpu {

typedef struct HypotOp {
public:
    // Hypot 是二元算子，计算 sqrt(x^2 + y^2)
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &x, const T &y) const {
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::hypot(x, y);
        } else {
            return static_cast<T>(std::hypot(static_cast<float>(x), static_cast<float>(y)));
        }
    }
} HypotOp;

} // namespace op::hypot::cpu

#endif // __HYPOT_CPU_H__
