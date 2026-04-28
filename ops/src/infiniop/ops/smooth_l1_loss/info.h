#ifndef __SMOOTH_L1_LOSS_INFO_H__
#define __SMOOTH_L1_LOSS_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::smooth_l1_loss {

class SmoothL1LossInfo {
    SmoothL1LossInfo() = default;

public:
    int _dtype;     // 数据类型 (float, half, etc.)
    size_t _numel;  // 参与计算的元素总数 (input.numel())
    float _beta;    // 平滑阈值参数
    int _reduction; // 规约模式 (0:None, 1:Mean, 2:Sum)

    int dtype() const { return _dtype; }
    size_t numel() const { return _numel; }
    float beta() const { return _beta; }
    int reduction() const { return _reduction; }

    static utils::Result<SmoothL1LossInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        float beta,
        int reduction) {

        // 1. 检查输入数据类型一致性 (Input vs Target)
        if (input_desc->dtype() != target_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. 检查输出数据类型一致性 (Output vs Input)
        if (out_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 3. 检查输入形状一致性 (Input vs Target)
        // SmoothL1Loss 要求 input 和 target 形状完全一致 (Elementwise)
        if (input_desc->ndim() != target_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &in_shape = input_desc->shape();
        const auto &tar_shape = target_desc->shape();
        size_t numel = input_desc->numel();

        for (size_t i = 0; i < input_desc->ndim(); ++i) {
            if (in_shape[i] != tar_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 4. 检查输出形状 (Output vs Reduction Mode)
        // Reduction枚举值: 0=None, 1=Mean, 2=Sum
        if (reduction == 0) {
            // Reduction::None -> 输出形状必须与输入一致
            if (out_desc->ndim() != input_desc->ndim()) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            const auto &out_shape = out_desc->shape();
            for (size_t i = 0; i < input_desc->ndim(); ++i) {
                if (out_shape[i] != in_shape[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        } else {
            // Reduction::Mean/Sum -> 输出通常是标量
            // 标量的定义可能是 ndim=0，或者 numel=1
            if (out_desc->numel() != 1) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // 5. 校验 beta 参数 (必须非负)
        if (beta < 0) {
            return INFINI_STATUS_BAD_PARAM;
        }
        return utils::Result<SmoothL1LossInfo>(SmoothL1LossInfo{
            input_desc->dtype(), // _dtype
            numel,               // _numel
            beta,                // _beta
            reduction            // _reduction
        });
    }
};

} // namespace op::smooth_l1_loss

#endif // __SMOOTH_L1_LOSS_INFO_H__
