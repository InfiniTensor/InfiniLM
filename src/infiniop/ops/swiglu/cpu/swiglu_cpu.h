#ifndef __SWIGLU_CPU_H__
#define __SWIGLU_CPU_H__

#include "../../../binary/cpu/binary_cpu.h"

BINARY_DESCRIPTOR(swiglu, cpu)

struct SwiGLUOp {
private:
    template <typename T>
    T sigmoid(const T &x) const {
        return 1 / (1 + std::exp(-x));
    }

public:
    template <typename T>
    T operator()(const T &up, const T &gate) const {
        return gate * sigmoid(gate) * up;
    }
};

#endif // __SWIGLU_CPU_H__
