#ifndef __SILU_CPU_H__
#define __SILU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(silu, cpu)

#include <cmath>

namespace op::silu::cpu {
typedef struct SiluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return x / (static_cast<T>(1) + std::exp(-x));
    }
} SiluOp;

} // namespace op::silu::cpu

#endif // __SILU_CPU_H__
