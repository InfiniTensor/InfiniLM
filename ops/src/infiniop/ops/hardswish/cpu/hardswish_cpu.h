#ifndef __HARDSWISH_CPU_H__
#define __HARDSWISH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(hardswish, cpu)

#include <algorithm>
#include <cmath>

namespace op::hardswish::cpu {

typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        const float x_f = utils::cast<float>(x);
        const float clamped = std::min(std::max(x_f + 3.0f, 0.0f), 6.0f);
        const float result = x_f * clamped * (1.0f / 6.0f);
        return utils::cast<T>(result);
    }
} HardSwishOp;

typedef struct HardSwishContiguousOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {

        T three = static_cast<T>(3);
        T zero = static_cast<T>(0);
        T six = static_cast<T>(6);

        T scale = static_cast<T>(0.16666667f);

        T val = x + three;

        val = std::max(zero, val);
        val = std::min(six, val);

        return x * val * scale;
    }
} HardSwishContiguousOp;

} // namespace op::hardswish::cpu

#endif
