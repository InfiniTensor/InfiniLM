#ifndef __TRIPLET_MARGIN_LOSS_INFO_H__
#define __TRIPLET_MARGIN_LOSS_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::triplet_margin_loss {

class TripletMarginLossInfo {
    TripletMarginLossInfo() = default;

public:
    int _dtype;     // 数据类型
    float _margin;  // 边界值
    int _p;         // 范数次数
    float _eps;     // 数值稳定性常数
    bool _swap;     // 是否交换距离
    int _reduction; // 规约模式 (0:None, 1:Mean, 2:Sum)

    // 形状信息缓存
    size_t _batch_size;  // N (样本数)
    size_t _feature_dim; // D (特征维度，即 input.numel() / N)

    int dtype() const { return _dtype; }
    float margin() const { return _margin; }
    int p() const { return _p; }
    float eps() const { return _eps; }
    bool swap() const { return _swap; }
    int reduction() const { return _reduction; }
    size_t batch_size() const { return _batch_size; }
    size_t feature_dim() const { return _feature_dim; }

    // 构造函数
    TripletMarginLossInfo(int dtype, float margin, int p, float eps, bool swap, int reduction,
                          size_t batch, size_t feature_dim)
        : _dtype(dtype), _margin(margin), _p(p), _eps(eps), _swap(swap), _reduction(reduction),
          _batch_size(batch), _feature_dim(feature_dim) {}

    static utils::Result<TripletMarginLossInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t anchor_desc,
        infiniopTensorDescriptor_t positive_desc,
        infiniopTensorDescriptor_t negative_desc,
        float margin,
        int p,
        float eps,
        int swap, // C 接口传入 int 替代 bool
        int reduction) {

        // 1. 检查输入形状一致性
        // Anchor, Positive, Negative 形状必须完全一致
        if (anchor_desc->ndim() != positive_desc->ndim() || anchor_desc->ndim() != negative_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t ndim = anchor_desc->ndim();
        for (size_t i = 0; i < ndim; ++i) {
            if (anchor_desc->shape()[i] != positive_desc->shape()[i] || anchor_desc->shape()[i] != negative_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 2. 检查数据类型
        // 所有输入和输出必须类型一致
        int dtype = anchor_desc->dtype();
        if (positive_desc->dtype() != dtype || negative_desc->dtype() != dtype || out_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        size_t N = 1;
        size_t D = 1;

        if (ndim > 0) {
            N = anchor_desc->shape()[0];
            for (size_t i = 1; i < ndim; ++i) {
                D *= anchor_desc->shape()[i];
            }
        } else {
            // 标量输入? 不太常见，暂且视为 N=1, D=1
            N = 1;
            D = 1;
        }

        // 4. 检查输出形状
        if (reduction == 0) { // None
            // 输出形状应为 (N)
            // 如果输入本身是 (N, D)，输出是 (N)
            // 严格检查：out ndim 应为 1 且 shape[0] == N
            if (out_desc->ndim() != 1 || out_desc->shape()[0] != N) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else { // Mean / Sum
            // 输出必须是标量
            if (out_desc->numel() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        return utils::Result<TripletMarginLossInfo>(TripletMarginLossInfo{
            dtype,
            margin,
            p,
            eps,
            static_cast<bool>(swap),
            reduction,
            N,
            D});
    }
};

} // namespace op::triplet_margin_loss

#endif // __TRIPLET_MARGIN_LOSS_INFO_H__
