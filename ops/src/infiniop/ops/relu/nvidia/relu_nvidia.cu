#ifdef ENABLE_NINETOOTHED
#include "../../../../../build/ninetoothed/relu.h"
#include "../../../ninetoothed/utils.h"
#endif
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "relu_nvidia.cuh"

namespace op::relu::nvidia {

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

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

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
#ifdef ENABLE_NINETOOTHED
    const auto &ndim{_info.getNdim()};

    auto x{ninetoothed::Tensor{inputs[0], _info.getInputShape(0), _info.getInputStrides(0), ndim}};
    auto y{ninetoothed::Tensor{output, _info.getOutputShape(), _info.getOutputStrides(), ndim}};

    constexpr auto block_size{1024};

    switch (_dtype) {
    case INFINI_DTYPE_F16:
    case INFINI_DTYPE_F32:
    case INFINI_DTYPE_F64:
    case INFINI_DTYPE_BF16:
        if (launch_relu(stream, x, y, ndim, _dtype, block_size)) {
            return INFINI_STATUS_INTERNAL_ERROR;
        }
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#else
    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::ReluOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::ReluOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::ReluOp, float>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::ReluOp, double>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#endif
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::relu::nvidia
