#include "bitwise_right_shift_cpu.h"

namespace op::bitwise_right_shift::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &input_desc = input_desc_vec.at(0);
    const auto &shift_desc = input_desc_vec.at(1);
    const auto &output_shape = out_desc->shape();
    const auto &input_shape = input_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64);

    if (input_desc->dtype() != dtype || shift_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_SAME_SHAPE(output_shape, input_shape);

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
    case INFINI_DTYPE_I8:
        return _device_info->calculate<BitwiseRightShiftOp, int8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I16:
        return _device_info->calculate<BitwiseRightShiftOp, int16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I32:
        return _device_info->calculate<BitwiseRightShiftOp, int32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I64:
        return _device_info->calculate<BitwiseRightShiftOp, int64_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U8:
        return _device_info->calculate<BitwiseRightShiftOp, uint8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U16:
        return _device_info->calculate<BitwiseRightShiftOp, uint16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U32:
        return _device_info->calculate<BitwiseRightShiftOp, uint32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U64:
        return _device_info->calculate<BitwiseRightShiftOp, uint64_t>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::bitwise_right_shift::cpu
