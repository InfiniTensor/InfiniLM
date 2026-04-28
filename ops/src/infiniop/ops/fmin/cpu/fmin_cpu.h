#ifndef __FMIN_CPU_H__
#define __FMIN_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(fmin, cpu)

namespace op::fmin::cpu {
typedef struct FminOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
            return utils::cast<T>(std::fminf(
                utils::cast<float>(a),
                utils::cast<float>(b)));
        } else if constexpr (std::is_floating_point_v<T>) {
            return std::fmin(a, b);
        } else {
            return std::min(a, b);
        }
    }
} FminOp;
} // namespace op::fmin::cpu

#endif // __FMIN_CPU_H__
