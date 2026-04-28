#ifndef __BROADCAST_TO_INFO_H__
#define __BROADCAST_TO_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <algorithm> // for std::max
#include <vector>

namespace op::broadcast_to {

class BroadcastToInfo {
    BroadcastToInfo() = default;

public:
    static constexpr int MAX_DIM = 8; // 定义最大维度，方便做定长数组

    int _dtype;
    int _ndim; // 统一后的维度（等于输出维度）
    size_t _count;

    // 存储对齐后的用于计算的信息
    int64_t _out_shape[MAX_DIM];
    int64_t _out_strides[MAX_DIM];
    int64_t _in_shape[MAX_DIM];   // 已经对齐并填充了1的输入Shape
    int64_t _in_strides[MAX_DIM]; // 已经处理过广播（stride=0）的输入Stride

    int dtype() const { return _dtype; }
    int ndim() const { return _ndim; }
    size_t count() const { return _count; }

    // 构造函数
    BroadcastToInfo(int dtype, int ndim, size_t count)
        : _dtype(dtype), _ndim(ndim), _count(count) {}

    static utils::Result<BroadcastToInfo> create(
        infiniopTensorDescriptor_t out_desc,
        const std::vector<infiniopTensorDescriptor_t> &input_descs) {

        if (input_descs.size() != 1) {
            return INFINI_STATUS_BAD_PARAM;
        }
        auto input_desc = input_descs[0];

        if (out_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->ndim() < input_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (out_desc->ndim() > MAX_DIM) {
            return INFINI_STATUS_BAD_PARAM;
        }

        BroadcastToInfo info(out_desc->dtype(), int(out_desc->ndim()), 0);

        // 1. 计算总元素个数并拷贝 Output 信息
        size_t count = 1;
        for (int i = 0; i < info._ndim; ++i) {
            info._out_shape[i] = out_desc->shape()[i];
            info._out_strides[i] = out_desc->strides()[i];
            count *= out_desc->shape()[i];
        }
        info._count = count;

        // 2. 维度对齐与广播规则检查 (Alignment & Broadcasting)
        // 计算维度差：例如 out(2,3,4), in(3,4) -> offset = 1
        int offset = info._ndim - int(input_desc->ndim());

        for (int i = 0; i < info._ndim; ++i) {
            // i 是输出的维度索引
            // in_i 是对应的输入维度索引
            int in_i = i - offset;

            int64_t out_dim = info._out_shape[i];
            int64_t in_dim = 1;    // 默认填充 1 (Input 维度不足时)
            int64_t in_stride = 0; // 默认 Stride 0 (对应填充的 1)

            if (in_i >= 0) {
                // 如果输入在这个维度有定义
                in_dim = input_desc->shape()[in_i];
                in_stride = input_desc->strides()[in_i];
            }

            // 检查规则
            if (in_dim != out_dim && in_dim != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }

            // 保存对齐后的信息
            info._in_shape[i] = in_dim;

            if (in_dim == 1 && out_dim > 1) {
                info._in_strides[i] = 0;
            } else {
                info._in_strides[i] = in_stride;
            }
        }

        return utils::Result<BroadcastToInfo>(info);
    }
};

} // namespace op::broadcast_to

#endif // __BROADCAST_TO_INFO_H__
