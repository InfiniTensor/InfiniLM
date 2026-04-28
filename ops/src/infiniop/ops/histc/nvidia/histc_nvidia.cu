#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"
#include "histc_nvidia.cuh"

namespace op::histc::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int64_t bins,
    double min_val,
    double max_val) {

    if (bins <= 0 || min_val >= max_val) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // Histc output is always float32. This backend also requires a contiguous output.
    if (y_desc->dtype() != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 1 || y_shape.size() != 1 || y_shape[0] != static_cast<size_t>(bins)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t input_size = x_shape[0];
    ptrdiff_t input_stride = x_desc->strides()[0];
    ptrdiff_t output_stride = y_desc->strides()[0];

    // This implementation treats y as a contiguous `float*` buffer.
    if (output_stride != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    // Negative (or broadcasted) strides are not supported by this kernel without an explicit base offset.
    if (input_stride <= 0) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(dtype, input_size, bins, min_val, max_val,
                               input_stride, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // Initialize output to zero
    CHECK_CUDA(cudaMemsetAsync(y, 0, _bins * sizeof(float), cuda_stream));

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (_input_size + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cuda::histc_kernel<half><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const half *>(x),
            _input_size,
            _input_stride,
            _bins,
            _min_val,
            _max_val);
        break;
    case INFINI_DTYPE_BF16:
        cuda::histc_kernel<cuda_bfloat16><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(x),
            _input_size,
            _input_stride,
            _bins,
            _min_val,
            _max_val);
        break;
    case INFINI_DTYPE_F32:
        cuda::histc_kernel<float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const float *>(x),
            _input_size,
            _input_stride,
            _bins,
            _min_val,
            _max_val);
        break;
    case INFINI_DTYPE_F64:
        cuda::histc_kernel<double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const double *>(x),
            _input_size,
            _input_stride,
            _bins,
            _min_val,
            _max_val);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::histc::nvidia
