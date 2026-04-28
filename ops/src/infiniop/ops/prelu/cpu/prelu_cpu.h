#ifndef __PRELU_CPU_H__
#define __PRELU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>
#include <cmath>

ELEMENTWISE_DESCRIPTOR(prelu, cpu)

namespace op::prelu::cpu {
typedef struct PreluOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &x, const T &weight) const {
        return x > 0 ? x : weight * x;
    }
} PreluOp;
} // namespace op::prelu::cpu

#endif // __PRELU_CPU_H__
