#include "all_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include <iostream>
namespace op::all::cpu {

Descriptor::~Descriptor() {}
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t *dim,
    size_t dim_size,
    bool keepdim) {
    auto result = AllInfo::create(output_desc, input_desc, dim, dim_size, keepdim);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {
template <typename Tdata>
infiniStatus_t calculateAll(
    const AllInfo &info,
    bool *output,
    const Tdata *input,
    size_t *dim,
    size_t dim_size,
    bool keepdim) {
    if (info.reduce_dim_size == info.ndim) {
        bool result = true;
        for (size_t index = 0; index < info.input_size; index++) {
            size_t input_offset = op::common_cpu::indexToOffset(index, info.ndim, info.permuted_input_shape.data(), info.permuted_input_strides.data());
            result = result && input[input_offset];
        }
        output[0] = result;
        return INFINI_STATUS_SUCCESS;
    } else {
        for (size_t i = info.output_size; i-- > 0;) {
            size_t output_offset = op::common_cpu::indexToOffset(i, info.output_shape.size(), info.output_shape.data(), info.output_strides.data());
            bool result = true;
            for (size_t j = 0; j < info.reduce_num; j++) {
                size_t input_flat = j + i * info.reduce_num;
                size_t input_offset = op::common_cpu::indexToOffset(input_flat, info.ndim, info.permuted_input_shape.data(), info.permuted_input_strides.data());
                Tdata input_val = input[input_offset];
                bool bool_val = static_cast<bool>(input_val);
                result = result && bool_val;
            }
            output[output_offset] = result;
        }
        return INFINI_STATUS_SUCCESS;
    }
}
} // namespace
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    size_t *dim,
    size_t dim_size,
    bool keepdim,
    void *stream) const {
    switch (_info.dtype) {
    case INFINI_DTYPE_BOOL:
        return calculateAll<bool>(_info, reinterpret_cast<bool *>(output), reinterpret_cast<const bool *>(input), dim, dim_size, keepdim);
    case INFINI_DTYPE_U8:
        return calculateAll<uint8_t>(_info, reinterpret_cast<bool *>(output), reinterpret_cast<const uint8_t *>(input), dim, dim_size, keepdim);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::all::cpu
