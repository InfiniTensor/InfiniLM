#ifndef __LOG10_CPU_H__
#define __LOG10_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(log10, cpu)

namespace op::log10::cpu {
typedef struct Log10Op {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::log10(x);
    }
} Log10Op;
} // namespace op::log10::cpu

#endif // __LOG10_CPU_H__
