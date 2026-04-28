#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"
#include "gaussian_nll_loss_nvidia.cuh"

namespace op::gaussian_nll_loss::nvidia {

Descriptor::~Descriptor() {
    if (reduce_buffer != nullptr) {
        cudaFree(reduce_buffer);
        reduce_buffer = nullptr;
    }
}

static bool build_meta(
    op::cuda::GaussianNllTensorMeta &meta,
    size_t ndim,
    const std::vector<size_t> &shape,
    const std::vector<ptrdiff_t> &strides) {

    if (ndim > static_cast<size_t>(op::cuda::kGaussianNllMaxDims)) {
        return false;
    }

    meta.ndim = static_cast<int>(ndim);
    for (size_t i = 0; i < static_cast<size_t>(op::cuda::kGaussianNllMaxDims); ++i) {
        meta.shape[i] = (i < ndim) ? shape[i] : 1;
        meta.strides[i] = (i < ndim) ? strides[i] : 0;
    }
    return true;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t var_desc,
    int full,
    double eps,
    int reduction) {

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto var_shape = var_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape || input_shape != var_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    Reduction red = static_cast<Reduction>(reduction);
    std::vector<size_t> expected_y_shape;
    if (red == Reduction::NONE) {
        expected_y_shape = input_shape;
    } else {
        expected_y_shape = {};
    }

    if (y_shape != expected_y_shape) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    void *reduce_buffer = nullptr;
    if (red != Reduction::NONE) {
        if (dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16) {
            CHECK_CUDA(cudaMalloc(&reduce_buffer, sizeof(float)));
        }
    }

    *desc_ptr = new Descriptor(
        dtype,
        input_desc->numel(),
        input_desc->ndim(),
        input_shape,
        y_desc->strides(),
        input_desc->strides(),
        target_desc->strides(),
        var_desc->strides(),
        full,
        eps,
        red,
        reduce_buffer,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *input,
    const void *target,
    const void *var,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    if (reduction == Reduction::NONE) {
        op::cuda::GaussianNllTensorMeta out_meta{};
        op::cuda::GaussianNllTensorMeta in_meta{};
        op::cuda::GaussianNllTensorMeta tgt_meta{};
        op::cuda::GaussianNllTensorMeta var_meta{};

        if (!build_meta(out_meta, ndim, shape, y_strides) || !build_meta(in_meta, ndim, shape, input_strides) || !build_meta(tgt_meta, ndim, shape, target_strides) || !build_meta(var_meta, ndim, shape, var_strides)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            float eps_val = static_cast<float>(eps);
            cuda::gaussian_nll_loss_kernel<half, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                reinterpret_cast<const half *>(var),
                input_size, out_meta, in_meta, tgt_meta, var_meta, eps_val, full);
            break;
        }
        case INFINI_DTYPE_BF16: {
            float eps_val = static_cast<float>(eps);
            cuda::gaussian_nll_loss_kernel<cuda_bfloat16, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(target),
                reinterpret_cast<const cuda_bfloat16 *>(var),
                input_size, out_meta, in_meta, tgt_meta, var_meta, eps_val, full);
            break;
        }
        case INFINI_DTYPE_F32: {
            float eps_val = static_cast<float>(eps);
            cuda::gaussian_nll_loss_kernel<float, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                reinterpret_cast<const float *>(var),
                input_size, out_meta, in_meta, tgt_meta, var_meta, eps_val, full);
            break;
        }
        case INFINI_DTYPE_F64: {
            double eps_val = eps;
            cuda::gaussian_nll_loss_kernel<double, double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                reinterpret_cast<const double *>(var),
                input_size, out_meta, in_meta, tgt_meta, var_meta, eps_val, full);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        // Sum or Mean reduction (scalar output)
        op::cuda::GaussianNllTensorMeta in_meta{};
        op::cuda::GaussianNllTensorMeta tgt_meta{};
        op::cuda::GaussianNllTensorMeta var_meta{};

        if (!build_meta(in_meta, ndim, shape, input_strides) || !build_meta(tgt_meta, ndim, shape, target_strides) || !build_meta(var_meta, ndim, shape, var_strides)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const bool is_mean = (reduction == Reduction::MEAN);
        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            float *accum = reinterpret_cast<float *>(reduce_buffer);
            if (accum == nullptr) {
                return INFINI_STATUS_INTERNAL_ERROR;
            }
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(float), cuda_stream));
            float eps_val = static_cast<float>(eps);

            const int reduce_blocks = std::min(num_blocks, 1024);
            cuda::gaussian_nll_loss_reduce_kernel<half, float><<<reduce_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                reinterpret_cast<const half *>(var),
                input_size, in_meta, tgt_meta, var_meta, eps_val, full);

            const float scale = is_mean ? (1.0f / static_cast<float>(input_size)) : 1.0f;
            cuda::gaussian_nll_loss_finalize_kernel<half, float><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y), accum, scale);
            break;
        }
        case INFINI_DTYPE_BF16: {
            float *accum = reinterpret_cast<float *>(reduce_buffer);
            if (accum == nullptr) {
                return INFINI_STATUS_INTERNAL_ERROR;
            }
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(float), cuda_stream));
            float eps_val = static_cast<float>(eps);

            const int reduce_blocks = std::min(num_blocks, 1024);
            cuda::gaussian_nll_loss_reduce_kernel<cuda_bfloat16, float><<<reduce_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(target),
                reinterpret_cast<const cuda_bfloat16 *>(var),
                input_size, in_meta, tgt_meta, var_meta, eps_val, full);

            const float scale = is_mean ? (1.0f / static_cast<float>(input_size)) : 1.0f;
            cuda::gaussian_nll_loss_finalize_kernel<cuda_bfloat16, float><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y), accum, scale);
            break;
        }
        case INFINI_DTYPE_F32: {
            float eps_val = static_cast<float>(eps);
            float *accum = reinterpret_cast<float *>(y);
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(float), cuda_stream));
            const int reduce_blocks = std::min(num_blocks, 1024);
            cuda::gaussian_nll_loss_reduce_kernel<float, float><<<reduce_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                reinterpret_cast<const float *>(var),
                input_size, in_meta, tgt_meta, var_meta, eps_val, full);
            const float scale = is_mean ? (1.0f / static_cast<float>(input_size)) : 1.0f;
            cuda::gaussian_nll_loss_finalize_kernel<float, float><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y), accum, scale);
            break;
        }
        case INFINI_DTYPE_F64: {
            double eps_val = eps;
            double *accum = reinterpret_cast<double *>(y);
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(double), cuda_stream));
            const int reduce_blocks = std::min(num_blocks, 1024);
            cuda::gaussian_nll_loss_reduce_kernel<double, double><<<reduce_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                reinterpret_cast<const double *>(var),
                input_size, in_meta, tgt_meta, var_meta, eps_val, full);
            const double scale = is_mean ? (1.0 / static_cast<double>(input_size)) : 1.0;
            cuda::gaussian_nll_loss_finalize_kernel<double, double><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y), accum, scale);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gaussian_nll_loss::nvidia
