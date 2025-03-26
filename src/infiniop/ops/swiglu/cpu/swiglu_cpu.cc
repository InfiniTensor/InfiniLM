#include "swiglu_cpu.h"

namespace op::swiglu::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t up_desc,
    infiniopTensorDescriptor_t gate_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    if (!SAME_VEC(out_shape, up_shape, gate_shape)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    op::binary::BinaryInfo info;
    CHECK_STATUS(op::binary::createBinaryInfo(info, out_desc, up_desc, gate_desc));

    // Create descriptor
    *desc_ptr = new Descriptor(
        dtype,
        std::move(info),
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *c,
    const void *a,
    const void *b,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        op::common_cpu::binary_op::calculate<fp16_t, SwiGLUOp>(_info, c, a, b);
        break;
    case INFINI_DTYPE_F32:
        op::common_cpu::binary_op::calculate<float, SwiGLUOp>(_info, c, a, b);
        break;
    case INFINI_DTYPE_F64:
        op::common_cpu::binary_op::calculate<double, SwiGLUOp>(_info, c, a, b);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::cpu
