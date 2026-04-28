#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "../../../reduce/cuda/reduce.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"
#include "layer_norm_nvidia.cuh"
#include <cub/block/block_reduce.cuh>

namespace op::layer_norm::nvidia {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata *output,
    Tdata *input_standardization,
    Tdata *input_std_deviation,
    const Tdata *input,
    const Tdata *weight,
    const Tdata *bias,
    float eps,
    size_t normalized_size,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_standardization_strides,
    const ptrdiff_t *input_std_deviation_strides,
    const ptrdiff_t *input_strides,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    bool bias_exist) {
    layerNormKernel<BLOCK_SIZE, Tdata, Tcompute>(
        output,
        input_standardization,
        input_std_deviation,
        input,
        weight,
        bias,
        eps,
        normalized_size,
        output_strides,
        input_standardization_strides,
        input_std_deviation_strides,
        input_strides,
        weight_stride,
        bias_stride,
        bias_exist);
}

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockLayernorm(
    Tdata *output,
    const Tdata *input,
    const Tdata *weight,
    const Tdata *bias,
    float eps,
    int dimsize,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    const size_t *shape,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    int ndim,
    bool bias_exist) {
    blockLayernormKernel<Tdata, BLOCK_SIZE>(output,
                                            input,
                                            weight,
                                            bias,
                                            eps,
                                            dimsize,
                                            output_strides,
                                            input_strides,
                                            shape,
                                            weight_stride,
                                            bias_stride,
                                            ndim,
                                            bias_exist);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpLayernorm(
    Tdata *output,
    const Tdata *input,
    const Tdata *weight,
    const Tdata *bias,
    float eps,
    int othersize,
    int dimsize,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    const size_t *shape,
    ptrdiff_t weight_stride,
    ptrdiff_t bias_stride,
    int ndim,
    bool bias_exist) {
    warpLayernormKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(output,
                                                           input,
                                                           weight,
                                                           bias,
                                                           eps,
                                                           othersize,
                                                           dimsize,
                                                           output_strides,
                                                           input_strides,
                                                           shape,
                                                           weight_stride,
                                                           bias_stride,
                                                           ndim,
                                                           bias_exist);
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_layer_norm(
    const LayerNormInfo &info,
    Tdata *output,
    Tdata *input_standardization,
    Tdata *input_std_deviation,
    const Tdata *input,
    const Tdata *weight,
    const Tdata *bias,
    cudaStream_t stream,
    void *workspace) {
    size_t ndim = info.ndim;
    char *workspace_ptr = reinterpret_cast<char *>(workspace);
    ptrdiff_t *input_strides_cuda = reinterpret_cast<ptrdiff_t *>(workspace_ptr);
    ptrdiff_t *output_strides_cuda = input_strides_cuda + ndim;
    ptrdiff_t *input_standardization_strides_cuda = output_strides_cuda + ndim;
    ptrdiff_t *input_std_deviation_strides_cuda = input_standardization_strides_cuda + ndim;

    size_t ptrdiff_array_size = 4 * ndim * sizeof(ptrdiff_t);
    size_t *shape_cuda = reinterpret_cast<size_t *>(workspace_ptr + ptrdiff_array_size);

    /// @todo: h2d copy breaks cuda graph, need to optimize this part in the future
    CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda, info.input_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_standardization_strides_cuda, info.input_standardization_strides.data(), sizeof(ptrdiff_t) * (ndim - 1), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_std_deviation_strides_cuda, info.input_std_deviation_strides.data(), sizeof(ptrdiff_t) * (ndim - 1), cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaMemcpyAsync(shape_cuda, info.input_shape.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream));
    int dimsize = (int)info.normalized_size;
    int num_blocks = (int)info.othersize;

    if (dimsize > 1024) {
        blockLayernorm<Tdata, BLOCK_SIZE>
            <<<num_blocks, BLOCK_SIZE, 0, stream>>>(output,
                                                    input,
                                                    weight,
                                                    bias,
                                                    info.eps,
                                                    dimsize,
                                                    output_strides_cuda,
                                                    input_strides_cuda,
                                                    shape_cuda,
                                                    info.weight_strides[0],
                                                    info.bias_exist ? info.bias_strides[0] : 0,
                                                    (int)info.ndim,
                                                    info.bias_exist);
    } else {
        constexpr unsigned int BLOCK_SIZE_x = 32;
        constexpr unsigned int BLOCK_SIZE_y = 32;

        int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
        dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);
        warpLayernorm<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
            <<<grid_dim, block_dim, 0, stream>>>(output,
                                                 input,
                                                 weight,
                                                 bias,
                                                 info.eps,
                                                 num_blocks,
                                                 dimsize,
                                                 output_strides_cuda,
                                                 input_strides_cuda,
                                                 shape_cuda,
                                                 info.weight_strides[0],
                                                 info.bias_exist ? info.bias_strides[0] : 0,
                                                 (int)info.ndim,
                                                 info.bias_exist);
    }

    return INFINI_STATUS_SUCCESS;
}

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float eps) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = output_desc->ndim() * (sizeof(ptrdiff_t) * 4 + sizeof(size_t));

    auto result = LayerNormInfo::createLayerNormInfo(
        output_desc,
        input_standardization_desc,
        input_std_deviation_desc,
        input_desc,
        weight_desc,
        bias_desc,
        eps);
    CHECK_RESULT(result);
    const LayerNormInfo &info = result.take();
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    void *input_standardization,
    void *input_std_deviation,
    const void *input,
    const void *weight,
    const void *bias,
    void *stream_) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

#define CALCULATE_LAYER_NORM(BLOCK_SIZE, TDATA) \
    calculate_layer_norm<BLOCK_SIZE, TDATA>(_info, (TDATA *)output, (TDATA *)input_standardization, (TDATA *)input_std_deviation, (const TDATA *)input, (const TDATA *)weight, (const TDATA *)bias, stream, workspace)
#define CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                               \
        if (_info.dtype == INFINI_DTYPE_F16)                        \
            return CALCULATE_LAYER_NORM(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                   \
            return CALCULATE_LAYER_NORM(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)                  \
            return CALCULATE_LAYER_NORM(BLOCK_SIZE, cuda_bfloat16); \
        else                                                        \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                  \
    }

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CALCULATE_LAYER_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_2048)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::layer_norm::nvidia
