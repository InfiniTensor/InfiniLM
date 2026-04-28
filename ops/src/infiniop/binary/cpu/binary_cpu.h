#ifndef __INFINIOP_BINARY_CPU_H__
#define __INFINIOP_BINARY_CPU_H__

#include "../../devices/cpu/common_cpu.h"
#include "../binary.h"
#include <utility>

namespace op::common_cpu {

namespace binary_op {

// Perform binary computation when inputs and the output can have different dtypes
template <typename Tc, typename Ta, typename Tb, typename BinaryOp, typename... Args>
void calculate(op::binary::BinaryInfo info, void *c, const void *a, const void *b, Args &&...args) {
    auto a_ = reinterpret_cast<const Ta *>(a);
    auto b_ = reinterpret_cast<const Tb *>(b);
    auto c_ = reinterpret_cast<Tc *>(c);
    ptrdiff_t data_size = info.c_data_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < data_size; ++i) {
        size_t a_index = info.contiguous ? i : op::common_cpu::indexToOffset(i, info.ndim, info.a_shape.data(), info.a_strides.data());
        size_t b_index = info.contiguous ? i : op::common_cpu::indexToOffset(i, info.ndim, info.b_shape.data(), info.b_strides.data());
        size_t c_index = info.contiguous ? i : (op::common_cpu::indexToOffset(i, info.ndim, info.c_shape.data(), info.c_strides.data()));

        c_[c_index] = BinaryOp{}(a_[a_index], b_[b_index], std::forward<Args>(args)...);
    }
}

// Perform binary computation when all inputs and the output share the same dtype
template <typename Tdata, typename BinaryOp, typename... Args>
void calculate(op::binary::BinaryInfo info, void *c, const void *a, const void *b, Args &&...args) {
    auto a_ = reinterpret_cast<const Tdata *>(a);
    auto b_ = reinterpret_cast<const Tdata *>(b);
    auto c_ = reinterpret_cast<Tdata *>(c);
    ptrdiff_t data_size = info.c_data_size;

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < data_size; ++i) {
        size_t a_index = info.contiguous ? i : op::common_cpu::indexToOffset(i, info.ndim, info.a_shape.data(), info.a_strides.data());
        size_t b_index = info.contiguous ? i : op::common_cpu::indexToOffset(i, info.ndim, info.b_shape.data(), info.b_strides.data());
        size_t c_index = info.contiguous ? i : (op::common_cpu::indexToOffset(i, info.ndim, info.c_shape.data(), info.c_strides.data()));

        if constexpr (std::is_same_v<Tdata, fp16_t>) {
            float a_val = utils::cast<float>(a_[a_index]);
            float b_val = utils::cast<float>(b_[b_index]);
            c_[c_index] = utils::cast<fp16_t>(BinaryOp{}(a_val, b_val, std::forward<Args>(args)...));
        } else {
            c_[c_index] = BinaryOp{}(a_[a_index], b_[b_index], std::forward<Args>(args)...);
        }
    }
}

} // namespace binary_op
} // namespace op::common_cpu

#endif // __INFINIOP_BINARY_CPU_H__
