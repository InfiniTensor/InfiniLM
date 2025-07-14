#ifndef __RELU_CPU_H__
#define __RELU_CPU_H__

#include <algorithm>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(relu, cpu)

namespace op::relu::cpu {
typedef struct ReluOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return std::max<T>(x, 0);
    }
} ReluOp;
} // namespace op::relu::cpu

#endif // __RELU_CPU_H__
