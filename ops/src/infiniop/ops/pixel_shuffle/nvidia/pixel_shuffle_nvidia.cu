#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"
#include "pixel_shuffle_nvidia.cuh"
#include <array>

namespace op::pixel_shuffle::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int upscale_factor) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    if (y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (upscale_factor <= 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    if (x_shape.size() != 4 || y_shape.size() != 4) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    if (x_desc->hasBroadcastDim() || y_desc->hasBroadcastDim()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    size_t batch = x_shape[0];
    size_t in_channels = x_shape[1];
    size_t height = x_shape[2];
    size_t width = x_shape[3];

    if (in_channels % (upscale_factor * upscale_factor) != 0) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t out_channels = in_channels / (upscale_factor * upscale_factor);
    size_t out_height = height * upscale_factor;
    size_t out_width = width * upscale_factor;

    std::vector<size_t> expected_y_shape = {batch, out_channels, out_height, out_width};
    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    std::array<ptrdiff_t, 4> x_strides = {x_desc->stride(0), x_desc->stride(1), x_desc->stride(2), x_desc->stride(3)};
    std::array<ptrdiff_t, 4> y_strides = {y_desc->stride(0), y_desc->stride(1), y_desc->stride(2), y_desc->stride(3)};

    *desc_ptr = new Descriptor(dtype, batch, in_channels, out_channels,
                               height, width, upscale_factor,
                               x_desc->numel(), y_desc->numel(),
                               x_strides, y_strides,
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    const size_t out_height = height * static_cast<size_t>(upscale_factor);
    const size_t out_width = width * static_cast<size_t>(upscale_factor);
    const size_t total = output_size;
    if (total == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    int num_blocks = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cuda::pixel_shuffle_kernel_strided<half><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(x),
            batch, out_channels, out_height, out_width,
            upscale_factor,
            x_strides[0], x_strides[1], x_strides[2], x_strides[3],
            y_strides[0], y_strides[1], y_strides[2], y_strides[3]);
        break;
    case INFINI_DTYPE_BF16:
        cuda::pixel_shuffle_kernel_strided<__nv_bfloat16><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<__nv_bfloat16 *>(y),
            reinterpret_cast<const __nv_bfloat16 *>(x),
            batch, out_channels, out_height, out_width,
            upscale_factor,
            x_strides[0], x_strides[1], x_strides[2], x_strides[3],
            y_strides[0], y_strides[1], y_strides[2], y_strides[3]);
        break;
    case INFINI_DTYPE_F32:
        cuda::pixel_shuffle_kernel_strided<float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const float *>(x),
            batch, out_channels, out_height, out_width,
            upscale_factor,
            x_strides[0], x_strides[1], x_strides[2], x_strides[3],
            y_strides[0], y_strides[1], y_strides[2], y_strides[3]);
        break;
    case INFINI_DTYPE_F64:
        cuda::pixel_shuffle_kernel_strided<double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<double *>(y),
            reinterpret_cast<const double *>(x),
            batch, out_channels, out_height, out_width,
            upscale_factor,
            x_strides[0], x_strides[1], x_strides[2], x_strides[3],
            y_strides[0], y_strides[1], y_strides[2], y_strides[3]);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::pixel_shuffle::nvidia
