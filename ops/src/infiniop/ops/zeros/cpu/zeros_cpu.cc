#include "zeros_cpu.h"

namespace op::zeros::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &x_desc = input_desc_vec.at(0);

    const auto &y_shape = out_desc->shape();
    const auto &x_shape = x_desc->shape();

    CHECK_DTYPE(dtype,
                INFINI_DTYPE_BYTE, // 1
                INFINI_DTYPE_BOOL, // 2
                INFINI_DTYPE_I8,   // 3
                INFINI_DTYPE_I16,  // 4
                INFINI_DTYPE_I32,  // 5
                INFINI_DTYPE_I64,  // 6
                INFINI_DTYPE_U8,   // 7
                INFINI_DTYPE_U16,  // 8
                INFINI_DTYPE_U32,  // 9
                INFINI_DTYPE_U64,  // 10
                INFINI_DTYPE_F8,   // 11
                INFINI_DTYPE_F16,  // 12
                INFINI_DTYPE_F32,  // 13
                INFINI_DTYPE_F64,  // 14
                INFINI_DTYPE_BF16, // 19
    );

    CHECK_SAME_SHAPE(y_shape, x_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_BYTE: // 1
        return _device_info->calculate<ZerosOp, uint8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_BOOL: // 2
        return _device_info->calculate<ZerosOp, bool>(_info, output, inputs, stream);
    case INFINI_DTYPE_I8: // 3
        return _device_info->calculate<ZerosOp, int8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I16: // 4
        return _device_info->calculate<ZerosOp, int16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I32: // 5
        return _device_info->calculate<ZerosOp, int32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_I64: // 6
        return _device_info->calculate<ZerosOp, int64_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U8: // 7
        return _device_info->calculate<ZerosOp, uint8_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U16: // 8
        return _device_info->calculate<ZerosOp, uint16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U32: // 9
        return _device_info->calculate<ZerosOp, uint32_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_U64: // 10
        return _device_info->calculate<ZerosOp, uint64_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F8: // 11
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_F16: // 12
        return _device_info->calculate<ZerosOp, fp16_t>(_info, output, inputs, stream);
    case INFINI_DTYPE_F32: // 13
        return _device_info->calculate<ZerosOp, float>(_info, output, inputs, stream);
    case INFINI_DTYPE_F64: // 14
        return _device_info->calculate<ZerosOp, double>(_info, output, inputs, stream);
    case INFINI_DTYPE_C16: // 15
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_C32: // 16
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_C64: // 17
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_C128: // 18
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_BF16: // 19
        return _device_info->calculate<ZerosOp, bf16_t>(_info, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::zeros::cpu
