#include "sum_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
namespace op::sum::cpu {

Descriptor::~Descriptor() {}
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t *dim,
    size_t dim_size,
    bool keepdim) {
    auto result = SumInfo::create(output_desc, input_desc, dim, dim_size, keepdim);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {
template <typename T>
infiniStatus_t calculateSum(
    const SumInfo *info,
    T *output,
    const T *input) {
    if (info->reduce_dim_size == info->permuted_input_shape.size()) { // 规约到标量
        float tempSum = 0.;
        for (size_t index = 0; index < info->input_size; index++) {
            size_t input_offset = op::common_cpu::indexToOffset(index, info->permuted_input_shape.size(), info->permuted_input_shape.data(), info->permuted_input_strides.data());
            tempSum += utils::cast<float>(input[input_offset]);
        }
        output[0] = utils::cast<T>(tempSum);
        return INFINI_STATUS_SUCCESS;
    } else {
        for (size_t i = 0; i < info->output_size; i++) {
            size_t output_offset = op::common_cpu::indexToOffset(i, info->output_shape.size(), info->output_shape.data(), info->output_strides.data());
            float tempSum = 0.;
            for (size_t j = 0; j < info->reduce_num; j++) {
                size_t input_offset = op::common_cpu::indexToOffset(j + i * info->reduce_num, info->permuted_input_shape.size(), info->permuted_input_shape.data(), info->permuted_input_strides.data());
                tempSum += utils::cast<float>(input[input_offset]);
            }
            output[output_offset] = utils::cast<T>(tempSum);
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
    void *stream) const {
    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return calculateSum<fp16_t>(&_info, (fp16_t *)output, reinterpret_cast<const fp16_t *>(input));
    case INFINI_DTYPE_F32:
        return calculateSum<float>(&_info, (float *)output, reinterpret_cast<const float *>(input));
    case INFINI_DTYPE_BF16:
        return calculateSum<bf16_t>(&_info, (bf16_t *)output, reinterpret_cast<const bf16_t *>(input));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::sum::cpu
