#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"
#include "kron_nvidia.cuh"

namespace op::kron::nvidia {

namespace {
constexpr size_t kMaxSupportedNdim = 8;
} // namespace

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto dtype = a_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    if (b_desc->dtype() != dtype || y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto a_shape = a_desc->shape();
    auto b_shape = b_desc->shape();
    auto y_shape = y_desc->shape();

    if (a_shape.size() != b_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t ndim = a_shape.size();
    if (ndim > kMaxSupportedNdim) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    std::vector<size_t> expected_y_shape(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        expected_y_shape[i] = a_shape[i] * b_shape[i];
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    auto a_strides = a_desc->strides();
    auto b_strides = b_desc->strides();
    auto y_strides = y_desc->strides();

    *desc_ptr = new Descriptor(dtype, ndim, a_shape, b_shape, y_shape,
                               a_strides, b_strides, y_strides,
                               a_desc->numel(), b_desc->numel(), y_desc->numel(),
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *a,
    const void *b,
    void *stream) const {

    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    size_t *a_shape_d = nullptr;
    size_t *b_shape_d = nullptr;
    size_t *y_shape_d = nullptr;
    ptrdiff_t *a_strides_d = nullptr;
    ptrdiff_t *b_strides_d = nullptr;
    ptrdiff_t *y_strides_d = nullptr;
    if (ndim > 0) {
        size_t *shape_data = reinterpret_cast<size_t *>(workspace);
        ptrdiff_t *stride_data = reinterpret_cast<ptrdiff_t *>(shape_data + 3 * ndim);
        a_shape_d = shape_data;
        b_shape_d = shape_data + ndim;
        y_shape_d = shape_data + 2 * ndim;
        a_strides_d = stride_data;
        b_strides_d = stride_data + ndim;
        y_strides_d = stride_data + 2 * ndim;

        CHECK_CUDA(cudaMemcpyAsync(a_shape_d, a_shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA(cudaMemcpyAsync(b_shape_d, b_shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA(cudaMemcpyAsync(y_shape_d, y_shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA(cudaMemcpyAsync(a_strides_d, a_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA(cudaMemcpyAsync(b_strides_d, b_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA(cudaMemcpyAsync(y_strides_d, y_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
    }

    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (y_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks < 1) {
        num_blocks = 1;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        cuda::kron_kernel<half><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(a),
            reinterpret_cast<const half *>(b),
            y_size, ndim,
            a_shape_d, b_shape_d, y_shape_d,
            a_strides_d, b_strides_d, y_strides_d);
        break;
    case INFINI_DTYPE_BF16:
        cuda::kron_kernel<cuda_bfloat16><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<cuda_bfloat16 *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(a),
            reinterpret_cast<const cuda_bfloat16 *>(b),
            y_size, ndim,
            a_shape_d, b_shape_d, y_shape_d,
            a_strides_d, b_strides_d, y_strides_d);
        break;
    case INFINI_DTYPE_F32:
        cuda::kron_kernel<float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<float *>(y),
            reinterpret_cast<const float *>(a),
            reinterpret_cast<const float *>(b),
            y_size, ndim,
            a_shape_d, b_shape_d, y_shape_d,
            a_strides_d, b_strides_d, y_strides_d);
        break;
    case INFINI_DTYPE_F64:
        cuda::kron_kernel<double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<double *>(y),
            reinterpret_cast<const double *>(a),
            reinterpret_cast<const double *>(b),
            y_size, ndim,
            a_shape_d, b_shape_d, y_shape_d,
            a_strides_d, b_strides_d, y_strides_d);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::kron::nvidia
