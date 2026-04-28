#ifndef __MULTI_MARGIN_LOSS_INFO_H__
#define __MULTI_MARGIN_LOSS_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::multi_margin_loss {

class MultiMarginLossInfo {
    MultiMarginLossInfo() = default;

public:
    int _dtype;          // 输入/权重/输出的数据类型
    int _p;              // 范数次数 (1 或 2)
    float _margin;       // 边界值
    int _reduction;      // 规约模式 (0:None, 1:Mean, 2:Sum)
    bool _has_weight;    // 是否存在权重张量
    size_t _batch_size;  // N
    size_t _num_classes; // C

    int dtype() const { return _dtype; }
    int p() const { return _p; }
    float margin() const { return _margin; }
    int reduction() const { return _reduction; }
    bool has_weight() const { return _has_weight; }
    size_t batch_size() const { return _batch_size; }
    size_t num_classes() const { return _num_classes; }

    // 构造函数
    MultiMarginLossInfo(int dtype, int p, float margin, int reduction, bool has_weight, size_t batch, size_t classes)
        : _dtype(dtype), _p(p), _margin(margin), _reduction(reduction),
          _has_weight(has_weight), _batch_size(batch), _num_classes(classes) {}

    static utils::Result<MultiMarginLossInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        infiniopTensorDescriptor_t weight_desc, // 可为 nullptr
        int p,
        float margin,
        int reduction) {

        // 1. 检查输入形状 (Input vs Target)
        // Input: (N, C), Target: (N)
        if (input_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (target_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t N = input_desc->shape()[0];
        size_t C = input_desc->shape()[1];

        if (target_desc->shape()[0] != N) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (target_desc->dtype() != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        // Output 和 Input 类型必须一致
        if (out_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        bool has_weight = (weight_desc != nullptr);
        if (has_weight) {
            // Weight: (C)
            if (weight_desc->ndim() != 1 || weight_desc->shape()[0] != C) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            // Weight 类型必须与 Input 一致
            if (weight_desc->dtype() != input_desc->dtype()) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        }
        if (reduction == 0) {
            if (out_desc->ndim() != 1 || out_desc->shape()[0] != N) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            // Reduction::Mean/Sum -> 输出必须是标量
            if (out_desc->numel() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        if (p != 1 && p != 2) {
            return INFINI_STATUS_BAD_PARAM;
        }
        return utils::Result<MultiMarginLossInfo>(MultiMarginLossInfo{
            input_desc->dtype(), // _dtype
            p,                   // _p
            margin,              // _margin
            reduction,           // _reduction
            has_weight,          // _has_weight
            N,                   // _batch_size
            C                    // _num_classes
        });
    }
};

} // namespace op::multi_margin_loss

#endif // __MULTI_MARGIN_LOSS_INFO_H__
