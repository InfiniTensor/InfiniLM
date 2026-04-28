#ifndef __ACOS_CPU_H__
#define __ACOS_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

// 使用宏声明 Descriptor 类
ELEMENTWISE_DESCRIPTOR(acos, cpu)

#include <cmath>
#include <type_traits>

namespace op::acos::cpu {

typedef struct AcosOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        if constexpr (std::is_integral_v<T>) {
            return static_cast<T>(std::acos(static_cast<double>(x)));
        } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::acos(x);
        } else {
            return static_cast<T>(std::acos(static_cast<float>(x)));
        }
    }
} AcosOp;

} // namespace op::acos::cpu

#endif // __ACOS_CPU_H__
