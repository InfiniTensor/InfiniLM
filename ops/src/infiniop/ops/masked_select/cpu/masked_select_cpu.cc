#include "masked_select_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::masked_select::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t mask_desc) {

    auto result = MaskedSelectInfo::create(input_desc, mask_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <typename T>
infiniStatus_t maskedSelect(const MaskedSelectInfo *info, const T *input, const bool *mask, void **data_ptr, size_t *dlen_ptr) {
    std::vector<T> res;
    for (size_t index = 0; index < info->total_elements; index++) {
        size_t input_offset = op::common_cpu::indexToOffset(index, info->ndim, info->shape.data(), info->input_strides.data());
        size_t mask_offset = op::common_cpu::indexToOffset(index, info->ndim, info->shape.data(), info->mask_strides.data());
        if (mask[mask_offset]) {
            res.push_back(input[input_offset]);
        }
    }
    *dlen_ptr = res.size();
    *data_ptr = new T[*dlen_ptr];
    std::memcpy(*data_ptr, res.data(), sizeof(T) * (*dlen_ptr));

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    const void *input,
    const bool *mask,
    void **data_ptr,
    size_t *dlen_ptr,
    void *stream) const {

    if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(maskedSelect(&_info, (const float *)input, mask, data_ptr, dlen_ptr));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::masked_select::cpu
