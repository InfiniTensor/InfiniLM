#ifndef __ERF_CPU_H__
#define __ERF_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(erf, cpu)

namespace op::erf::cpu {
typedef struct ErfOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::erf(x);
    }
} ErfOp;
} // namespace op::erf::cpu

#endif // __ERF_CPU_H__
