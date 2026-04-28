#ifndef __RELU6_CPU_H__
#define __RELU6_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>
#include <cmath>

ELEMENTWISE_DESCRIPTOR(relu6, cpu)

namespace op::relu6::cpu {
typedef struct Relu6Op {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::min(std::max(x, T(0)), T(6));
    }
} Relu6Op;
} // namespace op::relu6::cpu

#endif // __RELU6_CPU_H__
