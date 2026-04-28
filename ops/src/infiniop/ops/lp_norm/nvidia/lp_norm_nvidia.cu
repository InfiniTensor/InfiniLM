#include "../../../devices/nvidia/nvidia_common.cuh"
#include "lp_norm_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockLPNorm(
    Tdata *y, const Tdata *x,
    float p,
    size_t dimsize,
    ptrdiff_t stride, float eps) {
    blockLPNormKernel<Tdata, BLOCK_SIZE>(x, y, p, dimsize, stride, eps);
}

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockLPNormStrides(
    Tdata *y, const Tdata *x,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    const size_t *shape, int ndim, float p, size_t dimsize,
    float eps) {
    blockLPNormStridesKernel<Tdata, BLOCK_SIZE>(x, y, output_strides, input_strides, shape, ndim, p, dimsize, eps);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpLPNorm(
    Tdata *y, const Tdata *x,
    float p,
    size_t othersize,
    size_t dimsize,
    ptrdiff_t stride, float eps) {
    warpLPNormKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x, y, p, othersize, dimsize, stride, eps);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpLPNormStrides(
    Tdata *y, const Tdata *x,
    const ptrdiff_t *output_strides,
    const ptrdiff_t *input_strides,
    const size_t *shape, int ndim,
    float p, size_t othersize, size_t dimsize,
    float eps) {
    warpLPNormStridesKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x, y, output_strides, input_strides, shape, ndim, p, othersize, dimsize, eps);
}

namespace op::lp_norm::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis,
    int p,
    float eps) {
    auto info = LPNormInfo::createLPNormInfo(y_desc, x_desc, axis, p, eps);
    CHECK_RESULT(info);
    size_t workspace_size = y_desc->ndim() * (sizeof(ptrdiff_t) * 2 + sizeof(size_t));
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t launchKernel(const LPNormInfo &info, Tdata *y, const Tdata *x,
                            cudaStream_t stream, void *workspace) {
    size_t dimsize = info.dimsize;
    size_t othersize = info.othersize;
    float p_f = static_cast<float>(info.p);
    float eps = info.eps;
    int num_blocks = static_cast<float>(info.othersize);
    ptrdiff_t stride = info.stride;
    int ndim = (int)info.ndim;
    char *workspace_ptr = reinterpret_cast<char *>(workspace);
    ptrdiff_t *input_strides_cuda = reinterpret_cast<ptrdiff_t *>(workspace_ptr);
    ptrdiff_t *output_strides_cuda = input_strides_cuda + ndim;

    size_t ptrdiff_array_size = 2 * ndim * sizeof(ptrdiff_t);
    size_t *shape_cuda = reinterpret_cast<size_t *>(workspace_ptr + ptrdiff_array_size);
    CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda, info.input_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));

    CHECK_CUDA(cudaMemcpyAsync(shape_cuda, info.input_shape.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream));
    if (info.continuous) {
        if (dimsize > 1024) {
            blockLPNorm<Tdata, BLOCK_SIZE>
                <<<num_blocks, BLOCK_SIZE, 0, stream>>>(y, x,
                                                        p_f, dimsize, stride, eps);
        } else {
            constexpr unsigned int BLOCK_SIZE_x = 32;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpLPNorm<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                <<<grid_dim, block_dim, 0, stream>>>(y, x,
                                                     p_f, othersize, dimsize, stride, eps);
        }
    } else {
        if (info.axis == ndim - 1) {
            if (dimsize > 1024) {
                blockLPNormStrides<Tdata, BLOCK_SIZE>
                    <<<num_blocks, BLOCK_SIZE, 0, stream>>>(y, x, output_strides_cuda, input_strides_cuda, shape_cuda, ndim,
                                                            p_f, dimsize, eps);
            } else {
                constexpr unsigned int BLOCK_SIZE_x = 32;
                constexpr unsigned int BLOCK_SIZE_y = 32;
                int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
                dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
                dim3 grid_dim(num_block_x, 1, 1);
                warpLPNormStrides<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                    <<<grid_dim, block_dim, 0, stream>>>(y, x, output_strides_cuda, input_strides_cuda, shape_cuda, ndim,
                                                         p_f, othersize, dimsize, eps);
            }
        } else {
            return INFINI_STATUS_BAD_PARAM;
        }
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define CALCULATE_LP_NORM(BLOCK_SIZE, TDATA) \
    launchKernel<BLOCK_SIZE, TDATA>(_info, (TDATA *)y, (const TDATA *)x, stream, workspace)
#define CALCULATE_LP_NORM_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                            \
        if (_info.dtype == INFINI_DTYPE_F16)                     \
            return CALCULATE_LP_NORM(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                \
            return CALCULATE_LP_NORM(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)               \
            return CALCULATE_LP_NORM(BLOCK_SIZE, __nv_bfloat16); \
        else                                                     \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;               \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_LP_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_LP_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CALCULATE_LP_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_2048)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_LP_NORM_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::lp_norm::nvidia
