#include "swiglu_cpu.h"

namespace op::swiglu::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &up_desc = input_desc_vec.at(0);
    const auto &gate_desc = input_desc_vec.at(1);
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(out_shape, up_shape, gate_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<SwiGLUOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<SwiGLUOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<SwiGLUOp, double>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::cpu
