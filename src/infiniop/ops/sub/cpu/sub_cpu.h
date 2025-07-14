#ifndef __SUB_CPU_H__
#define __SUB_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(sub, cpu)

namespace op::sub::cpu {
typedef struct SubOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a - b;
    }
} SubOp;
} // namespace op::sub::cpu

#endif // __SUB_CPU_H__
