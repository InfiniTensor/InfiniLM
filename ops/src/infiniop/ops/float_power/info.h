#ifndef __FLOAT_POWER_INFO_H__
#define __FLOAT_POWER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::float_power {

class FloatPowerInfo {
    FloatPowerInfo() = default;

public:
    int _input_dtype;  // 输入数据类型
    int _output_dtype; // 输出数据类型

    bool _is_scalar_exponent; // 是否为标量指数
    float _scalar_exponent;   // 标量指数的值 (仅当 _is_scalar_exponent 为 true 时有效)

    size_t _num_elements; // 元素总数

    // Getters
    int input_dtype() const { return _input_dtype; }
    int output_dtype() const { return _output_dtype; }
    bool is_scalar_exponent() const { return _is_scalar_exponent; }
    float scalar_exponent() const { return _scalar_exponent; }
    size_t num_elements() const { return _num_elements; }

    // 构造函数
    FloatPowerInfo(int in_dtype, int out_dtype, bool is_scalar, float scalar_exp, size_t numel)
        : _input_dtype(in_dtype), _output_dtype(out_dtype),
          _is_scalar_exponent(is_scalar), _scalar_exponent(scalar_exp),
          _num_elements(numel) {}
    static utils::Result<FloatPowerInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t exponent_desc,
        float scalar_exponent) {
        if (out_desc->ndim() != input_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 使用引用接收 vector，避免之前的编译错误
        const auto &in_shape = input_desc->shape();
        const auto &out_shape = out_desc->shape();
        size_t count = 1;

        for (size_t i = 0; i < input_desc->ndim(); ++i) {
            if (in_shape[i] != out_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            count *= in_shape[i];
        }

        // 3. 判断是标量模式还是张量模式
        bool is_scalar = (exponent_desc == nullptr);

        if (!is_scalar) {
            if (exponent_desc->ndim() != input_desc->ndim()) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            const auto &exp_shape = exponent_desc->shape();
            for (size_t i = 0; i < input_desc->ndim(); ++i) {
                if (exp_shape[i] != in_shape[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        // 构造 Info 对象
        return utils::Result<FloatPowerInfo>(FloatPowerInfo{
            input_desc->dtype(), // Input Dtype
            out_desc->dtype(),   // Output Dtype (分开存储)
            is_scalar,           // Mode flag
            scalar_exponent,     // Scalar Value
            count                // Total elements
        });
    }
};

} // namespace op::float_power

#endif // __FLOAT_POWER_INFO_H__
