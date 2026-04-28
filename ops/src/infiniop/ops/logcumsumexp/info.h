#ifndef __LOGCUMSUMEXP_INFO_H__
#define __LOGCUMSUMEXP_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::logcumsumexp {

class LogCumSumExpInfo {
    LogCumSumExpInfo() = default;

public:
    int _dtype;
    int _axis;
    bool _exclusive;
    bool _reverse;

    size_t _outer_size;
    size_t _axis_size;
    size_t _inner_size;

    size_t _x_axis_stride;
    size_t _x_inner_stride;
    size_t _x_outer_stride;

    size_t _y_axis_stride;
    size_t _y_inner_stride;
    size_t _y_outer_stride;

    int dtype() const { return _dtype; }
    int axis() const { return _axis; }
    bool exclusive() const { return _exclusive; }
    bool reverse() const { return _reverse; }
    size_t outer_size() const { return _outer_size; }
    size_t axis_size() const { return _axis_size; }
    size_t inner_size() const { return _inner_size; }

    LogCumSumExpInfo(
        int dtype,
        int axis,
        bool exclusive,
        bool reverse,
        size_t outer,
        size_t axis_len,
        size_t inner,
        size_t x_as,
        size_t x_is,
        size_t x_os,
        size_t y_as,
        size_t y_is,
        size_t y_os)
        : _dtype(dtype),
          _axis(axis),
          _exclusive(exclusive),
          _reverse(reverse),
          _outer_size(outer),
          _axis_size(axis_len),
          _inner_size(inner),
          _x_axis_stride(x_as),
          _x_inner_stride(x_is),
          _x_outer_stride(x_os),
          _y_axis_stride(y_as),
          _y_inner_stride(y_is),
          _y_outer_stride(y_os) {}

    static utils::Result<LogCumSumExpInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int axis,
        int exclusive,
        int reverse) {

        if (y_desc->ndim() != x_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t ndim = x_desc->ndim();
        for (size_t i = 0; i < ndim; ++i) {
            if (y_desc->shape()[i] != x_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        if (y_desc->dtype() != x_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (axis < 0 || static_cast<size_t>(axis) >= ndim) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // ================================
        // 逻辑维度大小
        // ================================
        size_t outer = 1;
        for (size_t i = 0; i < static_cast<size_t>(axis); ++i) {
            outer *= x_desc->shape()[i];
        }

        size_t axis_len = x_desc->shape()[axis];

        size_t inner = 1;
        for (size_t i = static_cast<size_t>(axis) + 1; i < ndim; ++i) {
            inner *= x_desc->shape()[i];
        }

        // ================================
        // 物理 stride 计算 (已修复)
        // ================================

        // 1. Axis Stride
        size_t x_axis_stride = static_cast<size_t>(x_desc->stride(axis));
        size_t y_axis_stride = static_cast<size_t>(y_desc->stride(axis));

        // 2. Inner Stride
        // [关键修复]:
        // Kernel 将 inner 部分视为被展平的一维数组 (0 到 inner-1)。
        // 对于连续 (Contiguous) 的 Tensor，这部分数据的内存是连续的。
        // 因此，无论 inner 包含多少个逻辑维度，访问下一个元素的物理偏移量固定为 1。
        // 原先使用 stride(axis + 1) 会导致多维 inner 时跳过数据。
        size_t x_inner_stride = 1;
        size_t y_inner_stride = 1;

        // 3. Outer Stride
        // 如果 axis == 0，outer_size 为 1，outer stride 设为 0。
        // 否则取 axis 前一维的 stride，代表跨越整个 axis+inner 数据块的步长。
        size_t x_outer_stride = (axis == 0) ? 0 : static_cast<size_t>(x_desc->stride(axis - 1));
        size_t y_outer_stride = (axis == 0) ? 0 : static_cast<size_t>(y_desc->stride(axis - 1));

        return utils::Result<LogCumSumExpInfo>(LogCumSumExpInfo{
            x_desc->dtype(),
            axis,
            static_cast<bool>(exclusive),
            static_cast<bool>(reverse),
            outer,
            axis_len,
            inner,
            x_axis_stride,
            x_inner_stride,
            x_outer_stride,
            y_axis_stride,
            y_inner_stride,
            y_outer_stride});
    }
};

} // namespace op::logcumsumexp

#endif // __LOGCUMSUMEXP_INFO_H__
