#include "rearrange_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../../../tensor.h"

namespace op::rearrange::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = y_desc->dtype();
    auto ndim = y_desc->ndim();
    auto shape = y_desc->shape().data();

    CHECK_API_OR(x_desc->dtype(), dtype, return INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_API_OR(x_desc->ndim(), ndim, return INFINI_STATUS_BAD_TENSOR_SHAPE);

    for (size_t i = 0; i < ndim; ++i) {
        CHECK_API_OR(x_desc->shape()[i], shape[i], return INFINI_STATUS_BAD_TENSOR_SHAPE);
    }

    auto dst_strides = y_desc->strides().data();
    auto src_strides = x_desc->strides().data();
    auto element_size = infiniSizeOf(dtype);

    auto result = utils::RearrangeMeta::create(shape, dst_strides, src_strides, ndim, element_size);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        nullptr,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *y,
    const void *x,
    void *stream) const {
    _meta.launch(y, x);
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rearrange::cpu
