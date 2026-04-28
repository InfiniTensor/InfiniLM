#include "floor_moore.h"

#include "../../../elementwise/moore/elementwise_moore.h"

#include "floor_moore_kernel.h"

namespace op::floor::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // Floor is a unary operator, so we only look at the first input
    const auto &in_desc = input_desc_vec.at(0);
    const auto &out_shape = out_desc->shape();
    const auto &in_shape = in_desc->shape();

    // Floor supports floating point types generally, and int types (though effectively no-op)
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

    // Check if output shape matches input shape
    CHECK_SAME_SHAPE(out_shape, in_shape);

    // create MOORE elementwise descriptor
    CREATE_ELEMENTWISE_MOORE_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    // Use moore::FloorOp template
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::FloorOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, moore::FloorOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, moore::FloorOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, moore::FloorOp, double>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<256, moore::FloorOp, int32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<256, moore::FloorOp, int64_t>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::floor::moore
