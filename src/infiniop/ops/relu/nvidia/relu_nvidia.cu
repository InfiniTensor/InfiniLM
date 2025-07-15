#ifdef ENABLE_NINETOOTHED

#include "../../../../../build/ninetoothed/relu.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
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

    const auto &ndim{_info.getNdim()};
    const auto &x_shape_{_info.getInputShape(0)};
    const auto &x_strides_{_info.getInputStrides(0)};
    std::vector<uint64_t> x_shape_vec{x_shape_, x_shape_ + ndim};
    std::vector<int64_t> x_strides_vec{x_strides_, x_strides_ + ndim};
    auto x_data{const_cast<void *>(inputs[0])};
    auto x_shape{x_shape_vec.data()};
    auto x_strides{x_strides_vec.data()};
    const NineToothedTensor x{x_data, x_shape, x_strides};
    const auto &y_shape_{_info.getOutputShape()};
    const auto &y_strides_{_info.getOutputStrides()};
    std::vector<uint64_t> y_shape_vec{y_shape_, y_shape_ + ndim};
    std::vector<int64_t> y_strides_vec{y_strides_, y_strides_ + ndim};
    auto y_data{output};
    auto y_shape{y_shape_vec.data()};
    auto y_strides{y_strides_vec.data()};
    const NineToothedTensor y{y_data, y_shape, y_strides};
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

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::relu::nvidia

#endif
