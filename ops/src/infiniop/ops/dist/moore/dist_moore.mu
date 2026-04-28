#include "../../../devices/moore/moore_common.h"
#include "../../../devices/moore/moore_handle.h"
#include "../../../devices/moore/moore_kernel_common.h"
#include "../../../tensor.h"

#include "../cuda/kernel.cuh"
#include "dist_moore.h"

namespace op::dist::moore {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t x2_desc,
    double p) {

    auto dtype = x1_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto x1_shape = x1_desc->shape();
    auto x2_shape = x2_desc->shape();
    auto y_shape = y_desc->shape();

    if (x1_shape != x2_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    if (y_shape.size() != 0 && (y_shape.size() != 1 || y_shape[0] != 1)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const size_t ndim = x1_desc->ndim();
    if (ndim > static_cast<size_t>(DistIndexing::kMaxNdim)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    size_t input_size = x1_desc->numel();
    *desc_ptr = new Descriptor(dtype, input_size, p, ndim, x1_shape, x1_desc->strides(), x2_desc->strides(),
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x1,
    const void *x2,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<musaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;

    DistIndexing indexing{};
    indexing.ndim = static_cast<int>(_ndim);
    for (int d = 0; d < DistIndexing::kMaxNdim; ++d) {
        indexing.shape[d] = 1;
        indexing.x1_strides[d] = 0;
        indexing.x2_strides[d] = 0;
    }
    for (size_t d = 0; d < _ndim; ++d) {
        indexing.shape[d] = static_cast<int64_t>(_shape[d]);
        indexing.x1_strides[d] = static_cast<int64_t>(_x1_strides[d]);
        indexing.x2_strides[d] = static_cast<int64_t>(_x2_strides[d]);
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        cuda::dist_strided_out_kernel<BLOCK_SIZE, half, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<half *>(y),
            reinterpret_cast<const half *>(x1), reinterpret_cast<const half *>(x2),
            _input_size, _p, indexing);
        break;
    }
    case INFINI_DTYPE_BF16: {
        cuda::dist_strided_out_kernel<BLOCK_SIZE, cuda_bfloat16, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            reinterpret_cast<cuda_bfloat16 *>(y),
            reinterpret_cast<const cuda_bfloat16 *>(x1), reinterpret_cast<const cuda_bfloat16 *>(x2),
            _input_size, _p, indexing);
        break;
    }
    case INFINI_DTYPE_F32: {
        float *result_f = reinterpret_cast<float *>(y);
        cuda::dist_strided_kernel<BLOCK_SIZE, float, float><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_f, reinterpret_cast<const float *>(x1), reinterpret_cast<const float *>(x2),
            _input_size, _p, indexing);
        break;
    }
    case INFINI_DTYPE_F64: {
        double *result_d = reinterpret_cast<double *>(y);
        cuda::dist_strided_kernel<BLOCK_SIZE, double, double><<<1, BLOCK_SIZE, 0, cuda_stream>>>(
            result_d, reinterpret_cast<const double *>(x1), reinterpret_cast<const double *>(x2),
            _input_size, _p, indexing);
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dist::moore
