#ifndef __EQUAL_CPU_H__
#define __EQUAL_CPU_H__

#include <type_traits>

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(equal, cpu)

namespace op::equal::cpu {

typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;

    template <typename Tout, typename Tin0, typename Tin1>
    bool operator()(const Tin0 &a, const Tin1 &b) {
        if constexpr (std::is_same_v<Tin0, Tin1>) {
            return a == b;
        } else {
            return false;
        }
    }
} EqualOp;

} // namespace op::equal::cpu

#endif
