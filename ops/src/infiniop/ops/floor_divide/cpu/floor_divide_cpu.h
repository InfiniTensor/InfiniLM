#ifndef __FLOOR_DIVIDE_CPU_H__
#define __FLOOR_DIVIDE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>
#include <type_traits>

ELEMENTWISE_DESCRIPTOR(floor_divide, cpu)

namespace op::floor_divide::cpu {
typedef struct FloorDivideOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename T>
    T operator()(const T &a, const T &b) const {
        if constexpr (std::is_floating_point_v<T>) {
            return std::floor(a / b);
        } else {
            T res = a / b;
            T rem = a % b;
            if (rem != 0 && ((a < 0) ^ (b < 0))) {
                res -= 1;
            }
            return res;
        }
    }
} FloorDivideOp;
} // namespace op::floor_divide::cpu

#endif // __FLOOR_DIVIDE_CPU_H__
