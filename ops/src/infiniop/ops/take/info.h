#ifndef __TAKE_INFO_H__
#define __TAKE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::take {

class TakeInfo {
    TakeInfo() = default;

public:
    int _dtype;      // 数据类型 (float, half, etc.)
    int _idx_dtype;  // 索引类型 (int32, int64)
    size_t _num_out; // 输出元素总数 (== indices.numel())
    size_t _num_in;  // 输入元素总数 (用于边界检查)

    int dtype() const { return _dtype; }
    int idx_dtype() const { return _idx_dtype; }
    size_t num_out() const { return _num_out; }
    size_t num_in() const { return _num_in; }

    static utils::Result<TakeInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        infiniopTensorDescriptor_t indices_desc) {

        // 1. 检查数据类型一致性 (Output vs Input)
        if (out_desc->dtype() != in_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. 检查索引数据类型 (Indices)
        int idx_type = indices_desc->dtype();
        if (idx_type != INFINI_DTYPE_I32 && idx_type != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 3. 检查形状一致性 (Output vs Indices)
        if (out_desc->ndim() != indices_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &out_shape = out_desc->shape();
        const auto &idx_shape = indices_desc->shape();

        for (size_t i = 0; i < out_desc->ndim(); ++i) {
            if (out_shape[i] != idx_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 4. 提取信息
        int dtype = in_desc->dtype();
        size_t num_out = out_desc->numel();
        size_t num_in = in_desc->numel();
        return utils::Result<TakeInfo>(TakeInfo{
            dtype,
            idx_type,
            num_out,
            num_in});
    }
};

} // namespace op::take

#endif // __TAKE_INFO_H__
