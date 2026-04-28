#include "../../../../devices/nvidia/nvidia_common.cuh"
#include "per_channel_quant_int8_nvidia.cuh"

#include "../../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../../reduce/cuda/reduce.cuh"
#include <cub/block/block_reduce.cuh>

#include "../cuda/kernel.cuh"

template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockPerChannelQuantI8(
    int8_t *x_packed, float *x_scale, float *x_zero, const Tdata *x, int M, int K) {
    blockPerChannelQuantI8Kernel<Tdata, BLOCK_SIZE>(x_packed, x_scale, x_zero, x, M, K);
}
template <typename Tdata, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL blockPerChannelQuantI8Sym(
    int8_t *x_packed, float *x_scale, const Tdata *x, int M, int K) {
    blockPerChannelQuantI8SymKernel<Tdata, BLOCK_SIZE>(x_packed, x_scale, x, M, K);
}

template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpPerChannelQuantI8(
    int8_t *x_packed, float *x_scale, float *x_zero, const Tdata *x, int M, int K) {
    warpPerChannelQuantI8Kernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x_packed, x_scale, x_zero, x, M, K);
}
template <typename Tdata, unsigned int BLOCK_SIZE_x, unsigned int BLOCK_SIZE_y>
INFINIOP_CUDA_KERNEL warpPerChannelQuantI8Sym(
    int8_t *x_packed, float *x_scale, const Tdata *x, int M, int K) {
    warpPerChannelQuantI8SymKernel<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>(x_packed, x_scale, x, M, K);
}

namespace op::per_channel_quant_int8::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_packed_desc,
    infiniopTensorDescriptor_t x_scale_desc,
    infiniopTensorDescriptor_t x_zero_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto info = PerChannelQuantI8Info::createPerChannelQuantI8Info(x_packed_desc, x_scale_desc, x_zero_desc, x_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t per_channel_quant_int8Kernel(const PerChannelQuantI8Info &info, int8_t *x_packed, float *x_scale, float *x_zero, const Tdata *x, cudaStream_t stream) {
    int M = (int)info.M;
    int K = (int)info.K;

    if (K >= 1024) {
        if (x_zero == nullptr) {
            blockPerChannelQuantI8Sym<Tdata, BLOCK_SIZE>
                <<<M, BLOCK_SIZE, 0, stream>>>(x_packed, x_scale, x, M, K);
        } else {
            blockPerChannelQuantI8<Tdata, BLOCK_SIZE>
                <<<M, BLOCK_SIZE, 0, stream>>>(x_packed, x_scale, x_zero, x, M, K);
        }

    } else {
        constexpr unsigned int BLOCK_SIZE_x = 32;
        constexpr unsigned int BLOCK_SIZE_y = 32;
        int num_block_x = (M + BLOCK_SIZE_y - 1) / BLOCK_SIZE_y;
        dim3 block_dim(BLOCK_SIZE_x, BLOCK_SIZE_y, 1);
        dim3 grid_dim(num_block_x, 1, 1);
        if (x_zero == nullptr) {
            warpPerChannelQuantI8Sym<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                <<<grid_dim, block_dim, 0, stream>>>(x_packed, x_scale, x, M, K);
        } else {
            warpPerChannelQuantI8<Tdata, BLOCK_SIZE_x, BLOCK_SIZE_y>
                <<<grid_dim, block_dim, 0, stream>>>(x_packed, x_scale, x_zero, x, M, K);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *x_packed, void *x_scale, void *x_zero, const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define QUANT(BLOCK_SIZE, TDATA) \
    per_channel_quant_int8Kernel<BLOCK_SIZE, TDATA>(_info, (int8_t *)x_packed, (float *)x_scale, (float *)x_zero, (const TDATA *)x, stream)
#define QUANT_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                \
        if (_info.dtype == INFINI_DTYPE_F16)         \
            return QUANT(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)    \
            return QUANT(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)   \
            return QUANT(BLOCK_SIZE, __nv_bfloat16); \
        else                                         \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;   \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        QUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        QUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        QUANT_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::per_channel_quant_int8::nvidia
