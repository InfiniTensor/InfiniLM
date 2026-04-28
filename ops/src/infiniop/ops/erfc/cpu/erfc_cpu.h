#ifndef __ERFC_CPU_H__
#define __ERFC_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>

ELEMENTWISE_DESCRIPTOR(erfc, cpu)

namespace op::erfc::cpu {
typedef struct ErfcOp {
public:
    static constexpr size_t num_inputs = 1;
    template <typename T>
    T operator()(const T &x) const {
        return std::erfc(x);
    }
} ErfcOp;
} // namespace op::erfc::cpu

#endif // __ERFC_CPU_H__
