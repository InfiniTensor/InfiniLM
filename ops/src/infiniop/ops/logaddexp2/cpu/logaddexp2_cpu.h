#ifndef __LOGADDEXP2_CPU_H__
#define __LOGADDEXP2_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>
#include <cmath>

ELEMENTWISE_DESCRIPTOR(logaddexp2, cpu)

namespace op::logaddexp2::cpu {

typedef struct LogAddExp2Op {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &a, const T &b) const {
        if (a > b) {
            return a + std::log2(static_cast<T>(1) + std::exp2(b - a));
        } else {
            return b + std::log2(static_cast<T>(1) + std::exp2(a - b));
        }
    }
} LogAddExp2Op;

} // namespace op::logaddexp2::cpu

#endif // __LOGADDEXP2_CPU_H__
