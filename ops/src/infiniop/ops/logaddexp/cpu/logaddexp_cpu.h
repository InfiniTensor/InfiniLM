#ifndef __LOGADDEXP_CPU_H__
#define __LOGADDEXP_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <algorithm>
#include <cmath>

ELEMENTWISE_DESCRIPTOR(logaddexp, cpu)

namespace op::logaddexp::cpu {

typedef struct LogAddExpOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename T>
    T operator()(const T &a, const T &b) const {
        if (a > b) {
            return a + std::log(static_cast<T>(1) + std::exp(b - a));
        } else {
            return b + std::log(static_cast<T>(1) + std::exp(a - b));
        }
    }
} LogAddExpOp;

} // namespace op::logaddexp::cpu

#endif // __LOGADDEXP_CPU_H__
