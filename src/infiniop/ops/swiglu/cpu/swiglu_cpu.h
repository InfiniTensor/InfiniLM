#ifndef __SWIGLU_CPU_H__
#define __SWIGLU_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(swiglu, cpu)

namespace op::swiglu::cpu {
typedef struct SwiGLUOp {
private:
    template <typename T>
    T sigmoid(const T &x) const {
        return T(1) / (T(1) + std::exp(-x));
    }

public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &up, const T &gate) const {
        return gate * sigmoid(gate) * up;
    }
} SwiGLUOp;
} // namespace op::swiglu::cpu

#endif // __SWIGLU_CPU_H__
