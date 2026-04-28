#ifndef __RECIPROCAL_CPU_H__
#define __RECIPROCAL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(reciprocal, cpu)

namespace op::reciprocal::cpu {
typedef struct ReciprocalOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return static_cast<T>(1) / x;
    }
} ReciprocalOp;
} // namespace op::reciprocal::cpu

#endif // __RECIPROCAL_CPU_H__
