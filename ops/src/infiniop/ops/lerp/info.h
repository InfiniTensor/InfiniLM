#ifndef __LERP_INFO_H__
#define __LERP_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::lerp {

class LerpInfo {
    LerpInfo() = default;

public:
    int _dtype;             // 输入/输出的数据类型
    bool _is_scalar_weight; // 是否使用标量权重
    float _weight_scalar;   // 标量权重值 (当 _is_scalar_weight 为 true 时有效)
    size_t _numel;          // 输出元素总数

    int dtype() const { return _dtype; }
    bool is_scalar_weight() const { return _is_scalar_weight; }
    float weight_scalar() const { return _weight_scalar; }
    size_t numel() const { return _numel; }

    // 构造函数
    LerpInfo(int dtype, bool is_scalar_weight, float weight_scalar, size_t numel)
        : _dtype(dtype), _is_scalar_weight(is_scalar_weight),
          _weight_scalar(weight_scalar), _numel(numel) {}

    static utils::Result<LerpInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t start_desc,
        infiniopTensorDescriptor_t end_desc,
        infiniopTensorDescriptor_t weight_desc, // 如果为 nullptr，则启用标量模式
        float weight_scalar = 0.0f) {           // 标量模式下的权重值

        // 1. 基础指针检查
        if (out_desc == nullptr || start_desc == nullptr || end_desc == nullptr) {
            return INFINI_STATUS_BAD_PARAM;
        }

        // 2. 检查数据类型一致性
        // Lerp 要求 start, end, output 类型必须相同
        int dtype = start_desc->dtype();
        if (end_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (out_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 3. 处理权重模式 (Tensor vs Scalar)
        bool is_scalar = (weight_desc == nullptr);

        if (!is_scalar) {
            // Tensor 模式：仅检查 weight Tensor 的类型是否匹配
            if (weight_desc->dtype() != dtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        }
        // else: 标量模式，直接使用 weight_scalar

        // 4. 简单验证输出 (仅检查是否为空)
        // 按照要求，此处不进行 start/end/weight 之间的广播形状推导检查
        size_t numel = out_desc->numel();
        if (numel == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<LerpInfo>(LerpInfo{
            dtype,         // _dtype
            is_scalar,     // _is_scalar_weight
            weight_scalar, // _weight_scalar
            numel          // _numel
        });
    }
};

} // namespace op::lerp

#endif // __LERP_INFO_H__
