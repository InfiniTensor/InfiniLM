#include "../../../devices/nvidia/nvidia_common.cuh"
#include "softmax_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockSoftmax(
    Tdata *y, const Tdata *x,
    size_t dimsize,
    ptrdiff_t stride) {
    blockSoftmaxKernel<Tdata, BLOCK_SIZE>(x, y, dimsize, stride);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y, int numPerThreadx>
INFINIOP_CUDA_KERNEL warpSoftmax(
    Tdata *y, const Tdata *x,
    size_t othersize,
    size_t dimsize,
    ptrdiff_t stride) {
    warpSoftmaxKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y, numPerThreadx>(x, y, othersize, dimsize, stride);
}

namespace op::softmax::nvidia {

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
    int axis) {
    auto info = SoftmaxInfo::create(y_desc, x_desc, axis);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t dtype,
                            size_t othersize, size_t dimsize, ptrdiff_t stride,
                            cudaStream_t stream) {
    int num_blocks = (int)othersize;
    if (dtype == INFINI_DTYPE_F16) {
        if (dimsize > 1024) {
            blockSoftmax<half, BLOCK_SIZE>
                <<<num_blocks, BLOCK_SIZE, 0, stream>>>((half *)y, (const half *)x,
                                                        dimsize, stride);
        } else if (dimsize > 31) {
            constexpr unsigned int BLOCK_SIZE_x = 32;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            constexpr int numPerThreadx = 32;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpSoftmax<half, BLOCK_SIZE_x, BLOCK_SIZE_y, numPerThreadx>
                <<<grid_dim, block_dim, 0, stream>>>((half *)y, (const half *)x,
                                                     othersize, dimsize, stride);
        } else {
            constexpr unsigned int BLOCK_SIZE_x = 16;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            constexpr int numPerThreadx = 2;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpSoftmax<half, BLOCK_SIZE_x, BLOCK_SIZE_y, numPerThreadx>
                <<<grid_dim, block_dim, 0, stream>>>((half *)y, (const half *)x,
                                                     othersize, dimsize, stride);
        }

    } else if (dtype == INFINI_DTYPE_BF16) {
        if (dimsize > 1024) {
            blockSoftmax<cuda_bfloat16, BLOCK_SIZE>
                <<<num_blocks, BLOCK_SIZE, 0, stream>>>((cuda_bfloat16 *)y, (const cuda_bfloat16 *)x,
                                                        dimsize, stride);
        } else if (dimsize > 31) {
            constexpr unsigned int BLOCK_SIZE_x = 32;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            constexpr int numPerThreadx = 32;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpSoftmax<cuda_bfloat16, BLOCK_SIZE_x, BLOCK_SIZE_y, numPerThreadx>
                <<<grid_dim, block_dim, 0, stream>>>((cuda_bfloat16 *)y, (const cuda_bfloat16 *)x,
                                                     othersize, dimsize, stride);
        } else {
            constexpr unsigned int BLOCK_SIZE_x = 16;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            constexpr int numPerThreadx = 2;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpSoftmax<cuda_bfloat16, BLOCK_SIZE_x, BLOCK_SIZE_y, numPerThreadx>
                <<<grid_dim, block_dim, 0, stream>>>((cuda_bfloat16 *)y, (const cuda_bfloat16 *)x,
                                                     othersize, dimsize, stride);
        }

    } else if (dtype == INFINI_DTYPE_F32) {
        if (dimsize > 1024) {
            blockSoftmax<float, BLOCK_SIZE>
                <<<num_blocks, BLOCK_SIZE, 0, stream>>>((float *)y, (const float *)x,
                                                        dimsize, stride);
        } else if (dimsize > 31) {
            constexpr unsigned int BLOCK_SIZE_x = 32;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            constexpr int numPerThreadx = 32;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpSoftmax<float, BLOCK_SIZE_x, BLOCK_SIZE_y, numPerThreadx>
                <<<grid_dim, block_dim, 0, stream>>>((float *)y, (const float *)x,
                                                     othersize, dimsize, stride);
        } else {
            constexpr unsigned int BLOCK_SIZE_x = 16;
            constexpr unsigned int BLOCK_SIZE_y = 32;
            constexpr int numPerThreadx = 2;
            int num_block_x = (num_blocks + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
            dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
            dim3 grid_dim(num_block_x, 1, 1);
            warpSoftmax<float, BLOCK_SIZE_x, BLOCK_SIZE_y, numPerThreadx>
                <<<grid_dim, block_dim, 0, stream>>>((float *)y, (const float *)x,
                                                     othersize, dimsize, stride);
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            y, x, _info.dtype, _info.othersize, _info.dimsize, _info.stride, stream));

    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            y, x, _info.dtype, _info.othersize, _info.dimsize, _info.stride, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            y, x, _info.dtype, _info.othersize, _info.dimsize, _info.stride, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_2048>(
            y, x, _info.dtype, _info.othersize, _info.dimsize, _info.stride, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::softmax::nvidia
