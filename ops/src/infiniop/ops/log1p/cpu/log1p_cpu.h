#ifndef __LOG1P_CPU_H__
#define __LOG1P_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(log1p, cpu)

namespace op::log1p::cpu {
typedef struct Log1pOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::log1p(x);
    }
} Log1pOp;
} // namespace op::log1p::cpu

#endif // __LOG1P_CPU_H__
