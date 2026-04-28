#ifndef __ASINH_CPU_H__
#define __ASINH_CPU_H__

#include <cmath>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(asinh, cpu)

namespace op::asinh::cpu {
typedef struct AsinhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::asinh(x);
    }
} AsinhOp;
} // namespace op::asinh::cpu

#endif // __ASINH_CPU_H__
