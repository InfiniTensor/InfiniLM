#ifndef __FLOOR_CPU_H__
#define __FLOOR_CPU_H__

// 引入基础宏定义
#include "../../../elementwise/cpu/elementwise_cpu.h"

// 使用宏声明 Descriptor 类
ELEMENTWISE_DESCRIPTOR(floor, cpu)

#include <cmath>
#include <type_traits>

namespace op::floor::cpu {

typedef struct FloorOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &x) const {
        // 1. 整数类型：直接返回
        if constexpr (std::is_integral_v<T>) {
            return x;
        }
        // 2. 标准浮点类型 (float, double)：直接调用 std::floor，不降精度
        else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            return std::floor(x);
        }
        // 3. 半精度类型 (fp16, bf16)：先转 float 计算
        else {
            return static_cast<T>(std::floor(static_cast<float>(x)));
        }
    }
} FloorOp;

} // namespace op::floor::cpu

#endif // __FLOOR_CPU_H__
