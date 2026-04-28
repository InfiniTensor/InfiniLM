#include "softplus_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::softplus::cpu {

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
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    float beta,
    float threshold) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SoftplusInfo::create(out_desc, input_desc, beta, threshold);

    if (!result) {
        return result.status();
    }

    // 2. 创建 Descriptor
    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// -----------------------------------------------------------------------
// 核心计算实现：支持 Stride (非连续内存)
// -----------------------------------------------------------------------
template <typename T>
void calculate_cpu_impl(
    const SoftplusInfo &info,
    void *output,
    const void *input) {

    size_t num_elements = info.num_elements();
    float beta = info.beta();
    float threshold = info.threshold();

    // 获取内存布局信息 (依赖更新后的 SoftplusInfo)
    bool is_contiguous = info.is_contiguous();
    int ndim = info.ndim();
    const auto &shape = info.shape();
    const auto &strides = info.strides();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)num_elements; ++i) {
        // 1. 计算输入偏移量 (Input Offset)
        size_t input_offset = i; // 默认为线性索引

        if (!is_contiguous) {
            // 如果内存不连续，需要进行坐标变换：Linear Index -> Coordinate -> Physical Offset
            input_offset = 0;
            size_t temp_idx = i;
            for (int d = ndim - 1; d >= 0; --d) {
                size_t dim_size = shape[d];
                size_t coord = temp_idx % dim_size;
                temp_idx /= dim_size;
                input_offset += coord * strides[d];
            }
        }
        using CalcType = std::conditional_t<std::is_same_v<T, double>, double, float>;

        CalcType x = utils::cast<CalcType>(in_ptr[input_offset]);
        CalcType b = static_cast<CalcType>(beta);
        CalcType t = static_cast<CalcType>(threshold);

        CalcType bx = b * x;
        CalcType result;

        // 3. 计算 Softplus
        if (bx > t) {
            result = x;
        } else {
            result = std::log1p(std::exp(bx)) / b;
        }
        out_ptr[i] = utils::cast<T>(result);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        calculate_cpu_impl<float>(_info, output, input);
        break;
    case INFINI_DTYPE_F64:
        calculate_cpu_impl<double>(_info, output, input);
        break;
    case INFINI_DTYPE_F16:
        calculate_cpu_impl<fp16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_BF16:
        calculate_cpu_impl<bf16_t>(_info, output, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::softplus::cpu
