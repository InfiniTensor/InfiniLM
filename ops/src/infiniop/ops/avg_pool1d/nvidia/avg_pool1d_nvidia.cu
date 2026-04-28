#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "avg_pool1d_nvidia.cuh"

template <typename T>
__global__ void avgPool1dGlobalKernel(
    T *y,
    const T *x,
    size_t batch,
    size_t channels,
    size_t in_width,
    size_t out_width,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    ptrdiff_t y_stride_batch,
    ptrdiff_t y_stride_channel,
    ptrdiff_t y_stride_width,
    ptrdiff_t x_stride_batch,
    ptrdiff_t x_stride_channel,
    ptrdiff_t x_stride_width) {

    avgPool1dKernel<T>(
        y, x,
        batch, channels, in_width, out_width,
        kernel_size, stride, padding,
        y_stride_batch, y_stride_channel, y_stride_width,
        x_stride_batch, x_stride_channel, x_stride_width);
}

namespace op::avg_pool1d::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t kernel_size,
    size_t stride,
    size_t padding) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);

    auto info = AvgPool1dInfo::createAvgPool1dInfo(y_desc, x_desc, kernel_size, stride, padding);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t calculateAvgPool1d(
    const AvgPool1dInfo &info,
    int max_threads_per_block,
    T *y,
    const T *x,
    cudaStream_t stream) {

    size_t total_elements = info.batch * info.channels * info.out_width;

    int block_size = 256;
    if (max_threads_per_block > 0 && max_threads_per_block < 256) {
        block_size = max_threads_per_block;
    }

    size_t grid_size = (total_elements + block_size - 1) / block_size;
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    avgPool1dGlobalKernel<T><<<grid_size, block_size, 0, stream>>>(
        y, x,
        info.batch, info.channels, info.in_width, info.out_width,
        info.kernel_size, info.stride, info.padding,
        info.y_stride_batch, info.y_stride_channel, info.y_stride_width,
        info.x_stride_batch, info.x_stride_channel, info.x_stride_width);

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE(TDATA)                                        \
    calculateAvgPool1d(_info,                                   \
                       _opaque->internal->maxThreadsPerBlock(), \
                       (TDATA *)y,                              \
                       (const TDATA *)x,                        \
                       (cudaStream_t)stream)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return CALCULATE(half);
    case INFINI_DTYPE_BF16:
        return CALCULATE(cuda_bfloat16);
    case INFINI_DTYPE_F32:
        return CALCULATE(float);
    case INFINI_DTYPE_F64:
        return CALCULATE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE

} // namespace op::avg_pool1d::nvidia
