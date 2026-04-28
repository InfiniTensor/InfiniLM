#ifndef __ATANH_CPU_H__
#define __ATANH_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include <cmath>
#include <type_traits>

// 注册 atanh 算子在 cpu 后端的 descriptor
ELEMENTWISE_DESCRIPTOR(atanh, cpu)

namespace op::atanh::cpu {
typedef struct AtanhOp {
public:
    // atanh 是一元算子
    static constexpr size_t num_inputs = 1;

    template <typename T>
    T operator()(const T &a) const {
        // 对于 float, double 等原生支持的类型直接调用 std::atanh
        if constexpr (std::is_floating_point_v<T>) {
            return std::atanh(a);
        } else {
            // 对于 half, bfloat16 等自定义类型，先转为 float 计算再转回
            // 假设这些类型支持 static_cast 到 float
            return static_cast<T>(std::atanhf(static_cast<float>(a)));
        }
    }
} AtanhOp;
} // namespace op::atanh::cpu

#endif // __ATANH_CPU_H__
