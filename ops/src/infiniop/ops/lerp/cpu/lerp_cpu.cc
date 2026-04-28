#include "lerp_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::lerp::cpu {

// 【修改 1】定义一个独立的结构体来存储数据，避免 private 访问权限问题
struct LerpOpaqueData {
    int ndim;
    std::vector<size_t> output_shape;
    std::vector<int64_t> start_strides;
    std::vector<int64_t> end_strides;
    std::vector<int64_t> weight_strides;
};

// 【修改 2】Descriptor::Opaque 继承自 LerpOpaqueData
struct Descriptor::Opaque : public LerpOpaqueData {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

static std::vector<int64_t> compute_broadcast_strides(
    const std::vector<size_t> &out_shape,
    infiniopTensorDescriptor_t input_desc) {

    int out_ndim = static_cast<int>(out_shape.size());
    int in_ndim = static_cast<int>(input_desc->ndim());

    const auto &in_shape = input_desc->shape();
    const auto &in_strides = input_desc->strides();

    std::vector<int64_t> effective_strides(out_ndim, 0);

    for (int i = 0; i < out_ndim; ++i) {
        int out_idx = out_ndim - 1 - i;
        int in_idx = in_ndim - 1 - i;

        if (in_idx >= 0) {
            size_t dim_size = in_shape[in_idx];
            if (dim_size == 1) {
                effective_strides[out_idx] = 0;
            } else {
                effective_strides[out_idx] = in_strides[in_idx];
            }
        } else {
            effective_strides[out_idx] = 0;
        }
    }
    return effective_strides;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t start_desc,
    infiniopTensorDescriptor_t end_desc,
    infiniopTensorDescriptor_t weight_desc,
    float weight_scalar) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = LerpInfo::create(out_desc, start_desc, end_desc, weight_desc, weight_scalar);
    CHECK_RESULT(result);
    auto info = result.take();

    auto opaque = new Opaque();
    opaque->ndim = static_cast<int>(out_desc->ndim());
    opaque->output_shape = out_desc->shape();

    opaque->start_strides = compute_broadcast_strides(opaque->output_shape, start_desc);
    opaque->end_strides = compute_broadcast_strides(opaque->output_shape, end_desc);

    if (!info.is_scalar_weight() && weight_desc != nullptr) {
        opaque->weight_strides = compute_broadcast_strides(opaque->output_shape, weight_desc);
    }

    *desc_ptr = new Descriptor(
        opaque,
        info,
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 【修改 3】参数类型改为 const LerpOpaqueData *，这样外部函数可以访问
template <typename T>
void calculate_cpu_impl(
    const LerpInfo &info,
    const LerpOpaqueData *opaque,
    void *output,
    const void *start,
    const void *end,
    const void *weight) {

    size_t numel = info.numel();
    bool is_scalar_weight = info.is_scalar_weight();
    float scalar_w_val = info.weight_scalar();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto start_ptr = reinterpret_cast<const T *>(start);
    auto end_ptr = reinterpret_cast<const T *>(end);
    auto weight_ptr = is_scalar_weight ? nullptr : reinterpret_cast<const T *>(weight);

    int ndim = opaque->ndim;
    const auto &shape = opaque->output_shape;
    const auto &str_start = opaque->start_strides;
    const auto &str_end = opaque->end_strides;
    const auto &str_weight = opaque->weight_strides;

#pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)numel; ++i) {
        size_t temp_idx = i;
        int64_t offset_start = 0;
        int64_t offset_end = 0;
        int64_t offset_weight = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % shape[d];
            temp_idx /= shape[d];

            offset_start += coord * str_start[d];
            offset_end += coord * str_end[d];
            if (!is_scalar_weight) {
                offset_weight += coord * str_weight[d];
            }
        }

        T val_start = start_ptr[offset_start];
        T val_end = end_ptr[offset_end];

        T val_weight;
        if (is_scalar_weight) {
            val_weight = utils::cast<T>(scalar_w_val);
        } else {
            val_weight = weight_ptr[offset_weight];
        }

        float s = utils::cast<float>(val_start);
        float e = utils::cast<float>(val_end);
        float w = utils::cast<float>(val_weight);

        float res = s + w * (e - s);

        out_ptr[i] = utils::cast<T>(res);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *start,
    const void *end,
    const void *weight,
    void *stream) const {

    auto dtype = _info.dtype();

    // 在调用时，_opaque 会自动向上转型为 const LerpOpaqueData*
    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, _opaque, output, start, end, weight);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, _opaque, output, start, end, weight);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, _opaque, output, start, end, weight);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, _opaque, output, start, end, weight);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::lerp::cpu
