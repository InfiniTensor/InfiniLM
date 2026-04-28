#ifndef _FMOD_CPU_H__
#define _FMOD_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(fmod__, cpu)

namespace op::fmod__::cpu {
typedef struct FmodOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        return std::fmod(a, b);
    }
} FmodOp;
} // namespace op::fmod__::cpu

#endif // _FMOD_CPU_H__
