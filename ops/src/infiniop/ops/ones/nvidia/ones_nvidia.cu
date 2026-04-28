#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "ones_nvidia.cuh"

namespace op::ones::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
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

    // create CUDA elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

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

    switch (_dtype) {
    case INFINI_DTYPE_BYTE: // 1
        return _device_info->calculate<256, cuda::OnesOp, uint8_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BOOL: // 2
        return _device_info->calculate<256, cuda::OnesOp, bool>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I8: // 3
        return _device_info->calculate<256, cuda::OnesOp, int8_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I16: // 4
        return _device_info->calculate<256, cuda::OnesOp, int16_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I32: // 5
        return _device_info->calculate<256, cuda::OnesOp, int32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_I64: // 6
        return _device_info->calculate<256, cuda::OnesOp, int64_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U8: // 7
        return _device_info->calculate<256, cuda::OnesOp, uint8_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U16: // 8
        return _device_info->calculate<256, cuda::OnesOp, uint16_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U32: // 9
        return _device_info->calculate<256, cuda::OnesOp, uint32_t>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_U64: // 10
        return _device_info->calculate<256, cuda::OnesOp, uint64_t>(_info, workspace, output, inputs, stream);
#ifndef ENABLE_HYGON_API
    case INFINI_DTYPE_F8: // 11
        return _device_info->calculate<256, cuda::OnesOp, cuda_fp8_e4m3>(_info, workspace, output, inputs, stream);
#endif
    case INFINI_DTYPE_F16: // 12
        return _device_info->calculate<256, cuda::OnesOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32: // 13
        return _device_info->calculate<256, cuda::OnesOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64: // 14
        return _device_info->calculate<256, cuda::OnesOp, double>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_C16: // 15
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_C32: // 16
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_C64: // 17
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_C128: // 18
        return INFINI_STATUS_NOT_IMPLEMENTED;
    case INFINI_DTYPE_BF16: // 19
        return _device_info->calculate<256, cuda::OnesOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::ones::nvidia
