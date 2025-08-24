#ifndef __ADD_CPU_H__
#define __ADD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(add, cpu)

namespace op::add::cpu {
typedef struct AddOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return a + b;
    }
} AddOp;
} // namespace op::add::cpu

#endif // __ADD_CPU_H__
