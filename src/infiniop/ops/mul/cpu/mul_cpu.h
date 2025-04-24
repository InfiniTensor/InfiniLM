#ifndef __MUL_CPU_H__
#define __MUL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(mul, cpu)

namespace op::mul::cpu {
typedef struct MulOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a * b;
    }
} MulOp;
} // namespace op::mul::cpu

#endif // __MUL_CPU_H__
