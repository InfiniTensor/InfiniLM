#include "ldexp_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <type_traits>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::ldexp::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t exp_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 校验 shape 和广播规则
    auto result = LdexpInfo::create(y_desc, x_desc, exp_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (inputs.size() != 2) {
        return INFINI_STATUS_BAD_PARAM;
    }
    // 转发调用具体的参数接口
    return calculate(workspace, workspace_size, output, inputs[0], inputs[1], stream);
}

// ==================================================================
//                       计算核心逻辑实现
// ==================================================================

// 引入 TExp 模板参数，用于处理指数 exp 的不同数据类型
template <typename T, typename TExp>
void calculate_cpu_impl(
    const LdexpInfo &info,
    void *output,
    const void *x,
    const void *exp) {

    size_t total_tasks = info.count();

    // 获取广播所需的形状和步长信息
    int ndim = info.ndim();
    const auto &shape = info.shape();
    const auto &stride_x = info.x_strides();
    const auto &stride_exp = info.exp_strides();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto x_ptr = reinterpret_cast<const T *>(x);
    // 使用 TExp 转换指针，避免将 int 数据误读为 float
    auto exp_ptr = reinterpret_cast<const TExp *>(exp);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)total_tasks; ++i) {
        // 坐标映射逻辑：线性索引 -> 多维坐标 -> 输入偏移量
        size_t temp_idx = i;
        size_t offset_x = 0;
        size_t offset_exp = 0;

        // 从最低维开始反解坐标
        for (int d = ndim - 1; d >= 0; --d) {
            size_t dim_size = shape[d];
            size_t coord = temp_idx % dim_size;
            temp_idx /= dim_size;

            offset_x += coord * stride_x[d];
            offset_exp += coord * stride_exp[d];
        }

        float x_val;
        // 读取输入 x
        if constexpr (std::is_arithmetic_v<T>) {
            x_val = static_cast<float>(x_ptr[offset_x]);
        } else {
            x_val = utils::cast<float>(x_ptr[offset_x]);
        }

        // 读取 exp 并转换为 int
        int exp_int;
        if constexpr (std::is_arithmetic_v<TExp>) {
            exp_int = static_cast<int>(exp_ptr[offset_exp]);
        } else {
            // 如果 exp 是 fp16/bf16，先转 float 再转 int
            exp_int = static_cast<int>(utils::cast<float>(exp_ptr[offset_exp]));
        }

        // 计算 ldexp (x * 2^exp)
        float res = std::ldexp(x_val, exp_int);

        // 结果转回目标类型
        if constexpr (std::is_arithmetic_v<T>) {
            out_ptr[i] = static_cast<T>(res);
        } else {
            out_ptr[i] = utils::cast<T>(res);
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *x,
    const void *exp,
    void *stream) const {

    auto dtype = _info.dtype();
    auto exp_dtype = _info.exp_dtype();

    // 显式展开的双层 Switch 分发，每个 case 分 3 行
    switch (dtype) {
    case INFINI_DTYPE_F32:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            calculate_cpu_impl<float, int32_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_I64:
            calculate_cpu_impl<float, int64_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F32:
            calculate_cpu_impl<float, float>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F16:
            calculate_cpu_impl<float, fp16_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_BF16:
            calculate_cpu_impl<float, bf16_t>(_info, output, x, exp);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_F64:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            calculate_cpu_impl<double, int32_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_I64:
            calculate_cpu_impl<double, int64_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F32:
            calculate_cpu_impl<double, float>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F16:
            calculate_cpu_impl<double, fp16_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_BF16:
            calculate_cpu_impl<double, bf16_t>(_info, output, x, exp);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_F16:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            calculate_cpu_impl<fp16_t, int32_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_I64:
            calculate_cpu_impl<fp16_t, int64_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F32:
            calculate_cpu_impl<fp16_t, float>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F16:
            calculate_cpu_impl<fp16_t, fp16_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_BF16:
            calculate_cpu_impl<fp16_t, bf16_t>(_info, output, x, exp);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    case INFINI_DTYPE_BF16:
        switch (exp_dtype) {
        case INFINI_DTYPE_I32:
            calculate_cpu_impl<bf16_t, int32_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_I64:
            calculate_cpu_impl<bf16_t, int64_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F32:
            calculate_cpu_impl<bf16_t, float>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_F16:
            calculate_cpu_impl<bf16_t, fp16_t>(_info, output, x, exp);
            break;
        case INFINI_DTYPE_BF16:
            calculate_cpu_impl<bf16_t, bf16_t>(_info, output, x, exp);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::ldexp::cpu
