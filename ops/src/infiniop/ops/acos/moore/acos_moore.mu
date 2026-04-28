#include "acos_moore.h"

// 引入 Moore 平台的通用 Elementwise 描述符宏
#include "../../../elementwise/moore/elementwise_moore.h"

#include "acos_moore_kernel.h"

namespace op::acos::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    // Acos is a unary operator (y = acos(x))
    const auto &in_desc = input_desc_vec.at(0);
    const auto &out_shape = out_desc->shape();
    const auto &in_shape = in_desc->shape();

    // Acos supports floating point types.
    // Unlike floor, acos generally doesn't support integer outputs directly.
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // Check if output shape matches input shape
    CHECK_SAME_SHAPE(out_shape, in_shape);

    // create MOORE elementwise descriptor
    // 这里的宏会自动生成描述符初始化的通用代码
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

    // Use moore::AcosOp template defined in acos_moore_kernel.h
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, moore::AcosOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, moore::AcosOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, moore::AcosOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, moore::AcosOp, double>(_info, workspace, output, inputs, stream);

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::acos::moore
