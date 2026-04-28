#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "cross_entropy_nvidia.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tidx, typename Tcompute = float>
INFINIOP_CUDA_KERNEL crossEntropy(
    Tdata *y, const Tdata *x, const void *target,
    size_t outer_size, size_t vocab_size, ptrdiff_t x_stride) {

    crossEntropyKernel<BLOCK_SIZE, Tdata, Tidx, Tcompute>(
        y, x, target, outer_size, vocab_size, x_stride);
}

namespace op::cross_entropy::nvidia {

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
    infiniopTensorDescriptor_t target_desc) {

    auto x_dtype = x_desc->dtype();
    auto t_dtype = target_desc->dtype();

    CrossEntropyInfo info;
    info.dtype = x_dtype;
    info.target_dtype = t_dtype;

    info.vocab_size = x_desc->shape().back();
    info.outer_size = target_desc->numel();
    info.x_stride = static_cast<ptrdiff_t>(info.vocab_size);

    auto internal = reinterpret_cast<device::nvidia::Handle *>(handle)->internal();

    *desc_ptr = new Descriptor(
        new Opaque{internal},
        info, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, const void *target,
                            const CrossEntropyInfo &info, cudaStream_t stream) {

    dim3 grid(static_cast<uint32_t>(info.outer_size), 1, 1);

    if (info.target_dtype == INFINI_DTYPE_I64) {
        if (info.dtype == INFINI_DTYPE_F16) {
            crossEntropy<BLOCK_SIZE, half, int64_t>
                <<<grid, BLOCK_SIZE, 0, stream>>>((half *)y, (const half *)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_BF16) {
            crossEntropy<BLOCK_SIZE, __nv_bfloat16, int64_t>
                <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16 *)y, (const __nv_bfloat16 *)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_F32) {
            crossEntropy<BLOCK_SIZE, float, int64_t>
                <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const float *)x, target, info.outer_size, info.vocab_size, info.x_stride);
        }
    } else if (info.target_dtype == INFINI_DTYPE_I32) {

        if (info.dtype == INFINI_DTYPE_F16) {
            crossEntropy<BLOCK_SIZE, half, int32_t>
                <<<grid, BLOCK_SIZE, 0, stream>>>((half *)y, (const half *)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_BF16) {
            crossEntropy<BLOCK_SIZE, __nv_bfloat16, int32_t>
                <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16 *)y, (const __nv_bfloat16 *)x, target, info.outer_size, info.vocab_size, info.x_stride);
        } else if (info.dtype == INFINI_DTYPE_F32) {
            crossEntropy<BLOCK_SIZE, float, int32_t>
                <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const float *)x, target, info.outer_size, info.vocab_size, info.x_stride);
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     const void *target,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;

    int max_threads = _opaque->internal->maxThreadsPerBlock();

    if (max_threads >= 1024) {
        CHECK_STATUS(launchKernel<1024>(y, x, target, _info, stream));
    } else if (max_threads >= 512) {
        CHECK_STATUS(launchKernel<512>(y, x, target, _info, stream));
    } else {
        CHECK_STATUS(launchKernel<256>(y, x, target, _info, stream));
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::cross_entropy::nvidia
