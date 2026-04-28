#include "float_power_cpu.h"
#include "../../../../utils/custom_types.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

namespace op::float_power::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 创建描述符
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t exponent,
    float scalar_exponent) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 创建 Info 对象进行校验 (Info 类已更新，支持混合精度和 Tensor 指数)
    auto result = FloatPowerInfo::create(y, x, exponent, scalar_exponent);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        nullptr,
        result.take(),
        0, // CPU 不需要 workspace
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 核心计算逻辑
// 模板参数: T_OUT (输出类型), T_IN (输入类型)
// ==================================================================
template <typename T_OUT, typename T_IN>
void calculate_cpu_impl(
    const FloatPowerInfo &info,
    void *output,
    const void *input,
    const void *exponent_ptr) {

    size_t numel = info.num_elements();

    // 获取指数模式
    bool is_scalar = info.is_scalar_exponent();
    float scalar_exp = info.scalar_exponent();

    auto out_ptr = reinterpret_cast<T_OUT *>(output);
    auto in_ptr = reinterpret_cast<const T_IN *>(input);
    auto exp_ptr = reinterpret_cast<const T_IN *>(exponent_ptr);

    // 针对标量模式的简单优化标记
    bool is_square = is_scalar && (scalar_exp == 2.0f);
    bool is_sqrt = is_scalar && (scalar_exp == 0.5f);
    bool is_identity = is_scalar && (scalar_exp == 1.0f);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < ptrdiff_t(numel); ++i) {
        // 1. 读取输入并转为 float
        float in_val = utils::cast<float>(in_ptr[i]);
        float exp_val;

        // 2. 获取指数值
        if (is_scalar) {
            exp_val = scalar_exp;
        } else {
            // Tensor 模式：读取对应位置的指数并转为 float
            exp_val = utils::cast<float>(exp_ptr[i]);
        }

        // 3. 计算结果
        float result_val;
        if (is_scalar && is_identity) {
            result_val = in_val;
        } else if (is_scalar && is_square) {
            result_val = in_val * in_val;
        } else if (is_scalar && is_sqrt) {
            result_val = std::sqrt(in_val);
        } else {
            // 通用幂运算
            result_val = std::pow(in_val, exp_val);
        }

        // 4. 转回输出类型 T_OUT 并存储
        out_ptr[i] = utils::cast<T_OUT>(result_val);
    }
}

// ==================================================================
// 分发逻辑
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *exponent,
    void *stream) const {

    auto in_dtype = _info.input_dtype();
    auto out_dtype = _info.output_dtype();

// 定义内层宏：根据 Output 类型分发
#define DISPATCH_OUT(IN_T)                                                     \
    switch (out_dtype) {                                                       \
    case INFINI_DTYPE_F32:                                                     \
        cpu::calculate_cpu_impl<float, IN_T>(_info, output, input, exponent);  \
        return INFINI_STATUS_SUCCESS;                                          \
    case INFINI_DTYPE_F64:                                                     \
        cpu::calculate_cpu_impl<double, IN_T>(_info, output, input, exponent); \
        return INFINI_STATUS_SUCCESS;                                          \
    case INFINI_DTYPE_F16:                                                     \
        cpu::calculate_cpu_impl<fp16_t, IN_T>(_info, output, input, exponent); \
        return INFINI_STATUS_SUCCESS;                                          \
    case INFINI_DTYPE_BF16:                                                    \
        cpu::calculate_cpu_impl<bf16_t, IN_T>(_info, output, input, exponent); \
        return INFINI_STATUS_SUCCESS;                                          \
    default:                                                                   \
        return INFINI_STATUS_BAD_TENSOR_DTYPE;                                 \
    }

    // 外层 Switch：根据 Input 类型分发
    switch (in_dtype) {
    case INFINI_DTYPE_F32:
        DISPATCH_OUT(float);
    case INFINI_DTYPE_F64:
        DISPATCH_OUT(double);
    case INFINI_DTYPE_F16:
        DISPATCH_OUT(fp16_t);
    case INFINI_DTYPE_BF16:
        DISPATCH_OUT(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef DISPATCH_OUT
}

} // namespace op::float_power::cpu
