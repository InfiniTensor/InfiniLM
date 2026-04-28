#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../cuda/kernel.cuh"
#include "hinge_embedding_loss_nvidia.cuh"

namespace op::hinge_embedding_loss::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    double margin,
    int reduction) {

    auto dtype = input_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);
    if (target_desc->dtype() != dtype || y_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto input_shape = input_desc->shape();
    auto target_shape = target_desc->shape();
    auto y_shape = y_desc->shape();

    if (input_shape != target_shape) {
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

    const size_t ndim = input_desc->ndim();
    auto input_strides = input_desc->strides();
    auto target_strides = target_desc->strides();
    std::vector<ptrdiff_t> output_strides;
    if (red == Reduction::NONE) {
        output_strides = y_desc->strides();
    }

    *desc_ptr = new Descriptor(dtype, ndim, input_desc->numel(),
                               input_shape,
                               std::move(input_strides),
                               std::move(target_strides),
                               std::move(output_strides),
                               input_desc->isContiguous(),
                               target_desc->isContiguous(),
                               y_desc->isContiguous(),
                               margin, red,
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *input,
    const void *target,
    void *stream) const {

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = static_cast<int>((input_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    if (num_blocks < 1) {
        num_blocks = 1;
    }
    if (num_blocks > 1024) {
        num_blocks = 1024;
    }

    if (workspace_size < this->workspaceSize()) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (reduction == Reduction::NONE) {
        size_t *shape_d = nullptr;
        ptrdiff_t *input_strides_d = nullptr;
        ptrdiff_t *target_strides_d = nullptr;
        ptrdiff_t *output_strides_d = nullptr;
        if (ndim > 0) {
            shape_d = reinterpret_cast<size_t *>(workspace);
            input_strides_d = reinterpret_cast<ptrdiff_t *>(shape_d + ndim);
            target_strides_d = input_strides_d + ndim;
            output_strides_d = target_strides_d + ndim;

            CHECK_CUDA(cudaMemcpyAsync(shape_d, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, cuda_stream));
            CHECK_CUDA(cudaMemcpyAsync(input_strides_d, input_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
            CHECK_CUDA(cudaMemcpyAsync(target_strides_d, target_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
            CHECK_CUDA(cudaMemcpyAsync(output_strides_d, output_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
        }

        // Element-wise loss
        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            float margin_val = static_cast<float>(margin);
            cuda::hinge_embedding_loss_none_kernel<half, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y),
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                input_size,
                ndim,
                shape_d,
                output_strides_d,
                input_strides_d,
                target_strides_d,
                output_contiguous,
                input_contiguous,
                target_contiguous,
                margin_val);
            break;
        }
        case INFINI_DTYPE_BF16: {
            float margin_val = static_cast<float>(margin);
            cuda::hinge_embedding_loss_none_kernel<cuda_bfloat16, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y),
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(target),
                input_size,
                ndim,
                shape_d,
                output_strides_d,
                input_strides_d,
                target_strides_d,
                output_contiguous,
                input_contiguous,
                target_contiguous,
                margin_val);
            break;
        }
        case INFINI_DTYPE_F32: {
            float margin_val = static_cast<float>(margin);
            cuda::hinge_embedding_loss_none_kernel<float, float><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y),
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                input_size,
                ndim,
                shape_d,
                output_strides_d,
                input_strides_d,
                target_strides_d,
                output_contiguous,
                input_contiguous,
                target_contiguous,
                margin_val);
            break;
        }
        case INFINI_DTYPE_F64: {
            double margin_val = margin;
            cuda::hinge_embedding_loss_none_kernel<double, double><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y),
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                input_size,
                ndim,
                shape_d,
                output_strides_d,
                input_strides_d,
                target_strides_d,
                output_contiguous,
                input_contiguous,
                target_contiguous,
                margin_val);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        char *base = reinterpret_cast<char *>(workspace);
        char *meta = base + 8;
        size_t *shape_d = reinterpret_cast<size_t *>(meta);
        ptrdiff_t *input_strides_d = reinterpret_cast<ptrdiff_t *>(shape_d + ndim);
        ptrdiff_t *target_strides_d = input_strides_d + ndim;

        CHECK_CUDA(cudaMemcpyAsync(shape_d, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA(cudaMemcpyAsync(input_strides_d, input_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));
        CHECK_CUDA(cudaMemcpyAsync(target_strides_d, target_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice, cuda_stream));

        const bool mean = (reduction == Reduction::MEAN);

        // Sum or Mean: reduce into workspace accumulator, then write scalar output.
        switch (_dtype) {
        case INFINI_DTYPE_F16: {
            float margin_val = static_cast<float>(margin);
            float *accum = reinterpret_cast<float *>(base);
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(float), cuda_stream));
            cuda::hinge_embedding_loss_reduce_kernel<half, float, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const half *>(input),
                reinterpret_cast<const half *>(target),
                input_size,
                ndim,
                shape_d,
                input_strides_d,
                target_strides_d,
                input_contiguous,
                target_contiguous,
                margin_val);
            cuda::hinge_embedding_loss_finalize_kernel<half, float><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<half *>(y), accum, input_size, mean);
            break;
        }
        case INFINI_DTYPE_BF16: {
            float margin_val = static_cast<float>(margin);
            float *accum = reinterpret_cast<float *>(base);
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(float), cuda_stream));
            cuda::hinge_embedding_loss_reduce_kernel<cuda_bfloat16, float, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const cuda_bfloat16 *>(input),
                reinterpret_cast<const cuda_bfloat16 *>(target),
                input_size,
                ndim,
                shape_d,
                input_strides_d,
                target_strides_d,
                input_contiguous,
                target_contiguous,
                margin_val);
            cuda::hinge_embedding_loss_finalize_kernel<cuda_bfloat16, float><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<cuda_bfloat16 *>(y), accum, input_size, mean);
            break;
        }
        case INFINI_DTYPE_F32: {
            float margin_val = static_cast<float>(margin);
            float *accum = reinterpret_cast<float *>(base);
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(float), cuda_stream));
            cuda::hinge_embedding_loss_reduce_kernel<float, float, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const float *>(input),
                reinterpret_cast<const float *>(target),
                input_size,
                ndim,
                shape_d,
                input_strides_d,
                target_strides_d,
                input_contiguous,
                target_contiguous,
                margin_val);
            cuda::hinge_embedding_loss_finalize_kernel<float, float><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<float *>(y), accum, input_size, mean);
            break;
        }
        case INFINI_DTYPE_F64: {
            double margin_val = margin;
            double *accum = reinterpret_cast<double *>(base);
            CHECK_CUDA(cudaMemsetAsync(accum, 0, sizeof(double), cuda_stream));
            cuda::hinge_embedding_loss_reduce_kernel<double, double, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, cuda_stream>>>(
                accum,
                reinterpret_cast<const double *>(input),
                reinterpret_cast<const double *>(target),
                input_size,
                ndim,
                shape_d,
                input_strides_d,
                target_strides_d,
                input_contiguous,
                target_contiguous,
                margin_val);
            cuda::hinge_embedding_loss_finalize_kernel<double, double><<<1, 1, 0, cuda_stream>>>(
                reinterpret_cast<double *>(y), accum, input_size, mean);
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::hinge_embedding_loss::nvidia
