#include "vander_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

#include "../../../../utils/custom_types.h"

namespace op::vander::cpu {

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
    int N,
    int increasing) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 调用 Info::create 进行校验和元数据构建
    auto result = VanderInfo::create(out_desc, input_desc, N, increasing);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0, // CPU 实现通常不需要额外的 workspace
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void calculate_cpu_impl(
    const VanderInfo &info,
    void *output,
    const void *input) {

    size_t rows = info.rows(); // Input Size
    size_t cols = info.cols(); // Output Cols
    bool increasing = info.increasing();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

// 对每一行（输入向量的每个元素）进行并行计算
#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)rows; ++i) {
        // 将输入转换为 float/double 进行高精度计算，避免 fp16/bf16 累乘精度损失
        float x = utils::cast<float>(in_ptr[i]);

        // 优化：使用累乘法替代 pow 函数
        // x^0 = 1.0
        float val = 1.0f;

        if (increasing) {
            // 顺序：x^0, x^1, x^2 ...
            for (size_t j = 0; j < cols; ++j) {
                out_ptr[i * cols + j] = utils::cast<T>(val);
                val *= x;
            }
        } else {
            // 顺序：... x^2, x^1, x^0
            // 从最后一列 (x^0) 向前填充
            for (int64_t j = static_cast<int64_t>(cols) - 1; j >= 0; --j) {
                out_ptr[i * cols + j] = utils::cast<T>(val);
                val *= x;
            }
        }
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
        cpu::calculate_cpu_impl<float>(_info, output, input);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::vander::cpu
