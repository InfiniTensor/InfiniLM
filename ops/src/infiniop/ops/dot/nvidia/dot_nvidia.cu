#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../cuda/kernel.cuh"
#include "dot_nvidia.cuh"

namespace op::dot::nvidia {

__global__ void store_half_from_f32(half *dst, const float *src) {
    if (threadIdx.x == 0) {
        dst[0] = __float2half(src[0]);
    }
}

__global__ void store_bf16_from_f32(cuda_bfloat16 *dst, const float *src) {
    if (threadIdx.x == 0) {
        dst[0] = __float2bfloat16_rn(src[0]);
    }
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto dtype = a_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // This op does not do implicit dtype conversion: y/a/b must match.
    if (b_desc->dtype() != dtype || y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check that y is a scalar (0D tensor or shape [1])
    auto y_shape = y_desc->shape();
    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check that a and b are 1D vectors with same length
    auto a_shape = a_desc->shape();
    auto b_shape = b_desc->shape();
    if (a_shape.size() != 1 || b_shape.size() != 1 || a_shape[0] != b_shape[0]) {
        return INFINI_STATUS_BAD_PARAM;
    }

    size_t n = a_shape[0];
    ptrdiff_t a_stride = a_desc->strides()[0];
    ptrdiff_t b_stride = b_desc->strides()[0];

    // Negative/broadcasted strides are not supported without an explicit base offset.
    if (a_stride <= 0 || b_stride <= 0) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(dtype, n, a_stride, b_stride, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *a,
    const void *b,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr unsigned int BLOCK_SIZE = 256;

    switch (_dtype) {

    case INFINI_DTYPE_F32:
        cuda::dot_kernel<BLOCK_SIZE, float, float, float>
            <<<1, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(a),
                reinterpret_cast<const float *>(b),
                _n, _a_stride, _b_stride);
        break;

    case INFINI_DTYPE_F64:
        cuda::dot_kernel<BLOCK_SIZE, double, double, double>
            <<<1, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(a),
                reinterpret_cast<const double *>(b),
                _n, _a_stride, _b_stride);
        break;

    case INFINI_DTYPE_F16:
        cuda::dot_kernel<BLOCK_SIZE, half, half, float>
            <<<1, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(a),
                reinterpret_cast<const half *>(b),
                _n, _a_stride, _b_stride);
        break;

    case INFINI_DTYPE_BF16:
        cuda::dot_kernel<BLOCK_SIZE, cuda_bfloat16, cuda_bfloat16, float>
            <<<1, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(a),
                reinterpret_cast<const cuda_bfloat16 *>(b),
                _n, _a_stride, _b_stride);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dot::nvidia
