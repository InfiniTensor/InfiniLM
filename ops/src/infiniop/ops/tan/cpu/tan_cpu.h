#ifndef __TAN_CPU_H__
#define __TAN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(tan, cpu)

#include <cmath>

namespace op::tan::cpu {
typedef struct TanOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::tan(x);
    }
} TanOp;

} // namespace op::tan::cpu

#endif // __TAN_CPU_H__
