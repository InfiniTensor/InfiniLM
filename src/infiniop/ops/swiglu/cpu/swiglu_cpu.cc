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
    constexpr std::array<infiniDtype_t, 3> SUPPORTED_DTYPES = {
        INFINI_DTYPE_F16,
        INFINI_DTYPE_F32,
        INFINI_DTYPE_F64,
    };

    // Perform generic binary operator check
    CHECK_STATUS(op::common_cpu::binary_op::check(out_desc, up_desc, gate_desc, SUPPORTED_DTYPES, true, true));

    // Create descriptor
    *desc_ptr = new Descriptor(
        out_desc->dtype(),
        {out_desc, up_desc, gate_desc},
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
