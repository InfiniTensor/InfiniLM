#include "causal_softmax_cuda.cuh"

#include "../../../devices/cuda/cuda_common.cuh"
#include "causal_softmax_kernel.cuh"

namespace op::causal_softmax::cuda {

struct Descriptor::Opaque {
    std::shared_ptr<device::cuda::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc) {
    auto info = CausalSoftmaxInfo::create(y_desc);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::cuda::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *data, infiniDtype_t dtype, size_t batch_size, size_t seq_len, size_t total_seq_len, ptrdiff_t stride_b, ptrdiff_t stride_i, cudaStream_t stream) {
    dim3 grid(uint32_t(seq_len), uint32_t(batch_size), 1);
    if (dtype == INFINI_DTYPE_F16) {
        causalSoftmax<BLOCK_SIZE, half, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((half *)data, batch_size, seq_len, total_seq_len, stride_b, stride_i);
    } else if (dtype == INFINI_DTYPE_F32) {
        causalSoftmax<BLOCK_SIZE, float, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((float *)data, batch_size, seq_len, total_seq_len, stride_b, stride_i);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *data,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            data, _info.dtype, _info.batch_size, _info.seq_len, _info.total_seq_len, _info.stride_b, _info.stride_i, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            data, _info.dtype, _info.batch_size, _info.seq_len, _info.total_seq_len, _info.stride_b, _info.stride_i, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::cuda
