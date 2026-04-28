#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel_cuda.cuh"
#include "swiglu_nvidia_cuda.cuh"

template <typename T, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL SwiGLUCuda(
    T *c,
    const T *a,
    const T *b,
    int length,
    size_t batch, size_t seq_len, size_t hidden_dim,
    ptrdiff_t c_strides_0, ptrdiff_t c_strides_1, ptrdiff_t c_strides_2,
    ptrdiff_t a_strides_0, ptrdiff_t a_strides_1, ptrdiff_t a_strides_2,
    ptrdiff_t b_strides_0, ptrdiff_t b_strides_1, ptrdiff_t b_strides_2) {
    SwiGLUCudaKernel<T, BLOCK_SIZE>(c, a, b, length, batch, seq_len, hidden_dim,
                                    c_strides_0, c_strides_1, c_strides_2,
                                    a_strides_0, a_strides_1, a_strides_2,
                                    b_strides_0, b_strides_1, b_strides_2);
}

namespace op::swiglu_cuda::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

    auto info = SwiGLUCudaInfo::createSwiGLUCudaInfo(c_desc, a_desc, b_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename T>
infiniStatus_t calculate_swiglu_cuda(
    const SwiGLUCudaInfo &info,
    T *c,
    const T *a,
    const T *b,
    cudaStream_t stream,
    void *workspace) {

    int length = (int)info.length;
    size_t batch = info.batch;
    size_t seq_len = info.seq_len;
    size_t hidden_dim = info.hidden_dim;
    ptrdiff_t c_strides_0 = info.c_strides_0;
    ptrdiff_t c_strides_1 = info.c_strides_1;
    ptrdiff_t c_strides_2 = info.c_strides_2;
    ptrdiff_t a_strides_0 = info.a_strides_0;
    ptrdiff_t a_strides_1 = info.a_strides_1;
    ptrdiff_t a_strides_2 = info.a_strides_2;
    ptrdiff_t b_strides_0 = info.b_strides_0;
    ptrdiff_t b_strides_1 = info.b_strides_1;
    ptrdiff_t b_strides_2 = info.b_strides_2;

    int num_blocks = (length + BLOCK_SIZE - 1) / BLOCK_SIZE;
    SwiGLUCuda<T, BLOCK_SIZE>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(c, a, b, length, batch, seq_len, hidden_dim,
                                                c_strides_0, c_strides_1, c_strides_2,
                                                a_strides_0, a_strides_1, a_strides_2,
                                                b_strides_0, b_strides_1, b_strides_2);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream_) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    cudaStream_t stream = (cudaStream_t)stream_;

#define CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, TDATA) \
    calculate_swiglu_cuda<BLOCK_SIZE, TDATA>(_info, (TDATA *)c, (const TDATA *)a, (const TDATA *)b, stream, workspace)
#define CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                                \
        if (_info.dtype == INFINI_DTYPE_F16)                         \
            return CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                    \
            return CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)                   \
            return CALCULATE_SWIGLU_CUDA(BLOCK_SIZE, __nv_bfloat16); \
        else                                                         \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                   \
    }

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_2048)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_SWIGLU_CUDA_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu_cuda::nvidia
