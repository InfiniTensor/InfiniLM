#ifndef __SINH_CPU_H__
#define __SINH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(sinh, cpu)

namespace op::sinh::cpu {
typedef struct SinhOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::sinh(x);
    }
} SinhOp;
} // namespace op::sinh::cpu

#endif // __SINH_CPU_H__
