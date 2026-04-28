#ifndef __SIGMOID_CPU_H__
#define __SIGMOID_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(sigmoid, cpu)

namespace op::sigmoid::cpu {
typedef struct SigmoidOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return T(1) / (T(1) + std::exp(-x));
    }
} SigmoidOp;
} // namespace op::sigmoid::cpu

#endif // __SIGMOID_CPU_H__
