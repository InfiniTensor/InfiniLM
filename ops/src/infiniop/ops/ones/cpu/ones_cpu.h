#ifndef __ONES_CPU_H__
#define __ONES_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(ones, cpu)

namespace op::ones::cpu {
typedef struct OnesOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return static_cast<T>(1.0);
    }
} OnesOp;
} // namespace op::ones::cpu

#endif // __ONES_CPU_H__
