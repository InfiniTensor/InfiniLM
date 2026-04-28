#ifndef __GELU_CPU_H__
#define __GELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(gelu, cpu)

#include <cmath>

namespace op::gelu::cpu {
typedef struct GeluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return static_cast<T>(0.5 * x * (1 + erf(x / sqrt(2.0f))));
    }
} GeluOp;

} // namespace op::gelu::cpu

#endif // __GELU_CPU_H__
