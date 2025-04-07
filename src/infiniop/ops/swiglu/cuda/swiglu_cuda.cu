#include "swiglu_cuda.cuh"
#include "swiglu_cuda_internal.cuh"

namespace op::swiglu::cuda {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc) {

    auto handle = reinterpret_cast<device::cuda::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &up_desc = input_desc.at(0);
    const auto &gate_desc = input_desc.at(1);
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);
    if (!SAME_VEC(out_shape, up_shape, gate_shape)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // create CUDA elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        _device_info->calculate<256, SwiGLUOp, half>(_info, output, inputs, stream);
        break;
    case INFINI_DTYPE_F32:
        _device_info->calculate<256, SwiGLUOp, float>(_info, output, inputs, stream);
        break;
    case INFINI_DTYPE_F64:
        _device_info->calculate<256, SwiGLUOp, double>(_info, output, inputs, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::cuda
