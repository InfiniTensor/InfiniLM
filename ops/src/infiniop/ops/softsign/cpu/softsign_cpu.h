#ifndef __SOFTSIGN_CPU_H__
#define __SOFTSIGN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(softsign, cpu)

namespace op::softsign::cpu {
typedef struct SoftsignOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        return x / (static_cast<T>(1) + std::abs(x));
    }
} SoftsignOp;
} // namespace op::softsign::cpu

#endif // __SOFTSIGN_CPU_H__
