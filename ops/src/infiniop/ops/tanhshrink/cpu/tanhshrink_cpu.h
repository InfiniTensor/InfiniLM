#ifndef __TANHSHRINK_CPU_H__
#define __TANHSHRINK_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(tanhshrink, cpu)

#include <cmath>

namespace op::tanhshrink::cpu {
typedef struct TanhshrinkOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return x - tanh(x);
    }
} TanhshrinkOp;

} // namespace op::tanhshrink::cpu

#endif // __TANHSHRINK_CPU_H__
