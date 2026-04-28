#include "flipud_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cstdint>
#include <omp.h>
#include <vector>

// 引用框架定义的 float16/bfloat16 类型支持
#include "../../../../utils/custom_types.h"

namespace op::flipud::cpu {

// ==================================================================
// 0. 定义 Opaque 结构体
// ==================================================================
struct Descriptor::Opaque {
    std::vector<int64_t> shape;
    std::vector<int64_t> in_strides;
    std::vector<int64_t> out_strides;
    int ndim;
};

// ==================================================================
// 1. 析构函数
// ==================================================================
Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

// ==================================================================
// 2. 创建描述符
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 1. 创建 Info
    auto result = FlipudInfo::create(out_desc, input_desc);
    CHECK_RESULT(result);

    // 2. 创建并填充 Opaque
    auto opaque = new Descriptor::Opaque();
    opaque->ndim = static_cast<int>(input_desc->ndim());

    const auto &shape = input_desc->shape();
    const auto &in_strides = input_desc->strides();
    const auto &out_strides = out_desc->strides();

    for (int i = 0; i < opaque->ndim; ++i) {
        opaque->shape.push_back(shape[i]);
        opaque->in_strides.push_back(in_strides[i]);
        opaque->out_strides.push_back(out_strides[i]);
    }

    // 3. 创建 Descriptor
    *desc_ptr = new Descriptor(
        opaque,
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 3. 核心计算逻辑 implementation
// ==================================================================
// [修正] 直接接收具体参数，避开 Descriptor::Opaque 的私有权限问题
template <typename T>
void calculate_cpu_impl(
    int ndim,
    const std::vector<int64_t> &shape,
    const std::vector<int64_t> &in_strides,
    const std::vector<int64_t> &out_strides,
    size_t numel,
    void *output,
    const void *input) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    // 维度 0 的大小
    int64_t dim0_size = shape[0];

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)numel; ++i) {
        // --- A. 坐标反解 ---
        std::vector<int64_t> coords(ndim);

        size_t temp_idx = i;
        for (int d = ndim - 1; d >= 0; --d) {
            coords[d] = temp_idx % shape[d];
            temp_idx /= shape[d];
        }

        // --- B. 计算输出偏移量 ---
        size_t out_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            out_offset += coords[d] * out_strides[d];
        }

        // --- C. 翻转逻辑 (Flip Axis 0) ---
        coords[0] = dim0_size - 1 - coords[0];

        // --- D. 计算输入偏移量 ---
        size_t in_offset = 0;
        for (int d = 0; d < ndim; ++d) {
            in_offset += coords[d] * in_strides[d];
        }

        // --- E. 数据搬运 ---
        out_ptr[out_offset] = in_ptr[in_offset];
    }
}

// ==================================================================
// 4. 执行计算 (Calculate 分发)
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();
    size_t numel = _info.numel();

    // 显式 Switch-Case 分发
    // 在这里解包 _opaque，因为 calculate 是成员函数，可以访问 private 的 _opaque
    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(
            _opaque->ndim, _opaque->shape, _opaque->in_strides, _opaque->out_strides,
            numel, output, input);
        break;

    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(
            _opaque->ndim, _opaque->shape, _opaque->in_strides, _opaque->out_strides,
            numel, output, input);
        break;

    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(
            _opaque->ndim, _opaque->shape, _opaque->in_strides, _opaque->out_strides,
            numel, output, input);
        break;

    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(
            _opaque->ndim, _opaque->shape, _opaque->in_strides, _opaque->out_strides,
            numel, output, input);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::flipud::cpu
