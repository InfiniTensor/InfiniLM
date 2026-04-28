#include "../../../devices/moore/moore_common.h"
#include "adaptive_max_pool1d_moore.h"

#include "../../../devices/moore/moore_kernel_common.h"

#include "../cuda/kernel.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_MOORE_KERNEL adaptiveMaxPool1dKernel(
    Tdata *__restrict__ y,
    ptrdiff_t stride_y_batch,
    ptrdiff_t stride_y_channel,
    const Tdata *__restrict__ x,
    ptrdiff_t stride_x_batch,
    ptrdiff_t stride_x_channel,
    ptrdiff_t stride_x_length,
    size_t channels,
    size_t input_length,
    size_t output_length,
    size_t ndim) {

    adaptiveMaxPool1dBlock<BLOCK_SIZE, Tdata, Tcompute>(
        y, stride_y_batch, stride_y_channel,
        x, stride_x_batch, stride_x_channel, stride_x_length,
        channels, input_length, output_length, ndim);
}

namespace op::adaptive_max_pool1d::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t output_size) {
    auto result = AdaptiveMaxPool1dInfo::create(y_desc, x_desc, output_size);
    CHECK_RESULT(result);
    auto info = result.take();

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::moore::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(
    uint32_t num_blocks,
    void *y, infiniDtype_t dtype,
    ptrdiff_t stride_y_batch, ptrdiff_t stride_y_channel,
    const void *x,
    ptrdiff_t stride_x_batch, ptrdiff_t stride_x_channel, ptrdiff_t stride_x_length,
    size_t channels, size_t input_length, size_t output_length, size_t ndim,
    musaStream_t musa_stream) {

#define LAUNCH_KERNEL(Tdata, Tcompute)                                                                \
    adaptiveMaxPool1dKernel<BLOCK_SIZE, Tdata, Tcompute><<<num_blocks, BLOCK_SIZE, 0, musa_stream>>>( \
        reinterpret_cast<Tdata *>(y),                                                                 \
        stride_y_batch, stride_y_channel,                                                             \
        reinterpret_cast<const Tdata *>(x),                                                           \
        stride_x_batch, stride_x_channel, stride_x_length,                                            \
        channels, input_length, output_length, ndim)

    if (dtype == INFINI_DTYPE_F16) {
        LAUNCH_KERNEL(half, float);
    } else if (dtype == INFINI_DTYPE_BF16) {
        LAUNCH_KERNEL(__mt_bfloat16, float);
    } else if (dtype == INFINI_DTYPE_F32) {
        LAUNCH_KERNEL(float, float);
    } else if (dtype == INFINI_DTYPE_F64) {
        LAUNCH_KERNEL(double, double);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_KERNEL

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    const size_t ndim = _info.ndim();
    const size_t batch_size = _info.shape[0];
    const size_t channels = ndim > 2 ? _info.shape[1] : 1;
    const size_t input_length = _info.input_length();
    const size_t output_length = _info.output_length();

    ptrdiff_t stride_x_batch = _info.x_strides[0];
    ptrdiff_t stride_x_channel = ndim > 2 ? _info.x_strides[1] : 0;
    ptrdiff_t stride_x_length = _info.x_strides.back();

    ptrdiff_t stride_y_batch = _info.y_strides[0];
    ptrdiff_t stride_y_channel = ndim > 2 ? _info.y_strides[1] : 0;

    uint32_t num_blocks = static_cast<uint32_t>(batch_size * channels);
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    if (_opaque->internal->maxThreadsPerBlock() >= MOORE_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<MOORE_BLOCK_SIZE_1024>(
            num_blocks, y, _info.atype,
            stride_y_batch, stride_y_channel,
            x, stride_x_batch, stride_x_channel, stride_x_length,
            channels, input_length, output_length, ndim,
            musa_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() >= MOORE_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<MOORE_BLOCK_SIZE_512>(
            num_blocks, y, _info.atype,
            stride_y_batch, stride_y_channel,
            x, stride_x_batch, stride_x_channel, stride_x_length,
            channels, input_length, output_length, ndim,
            musa_stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == MOORE_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<MOORE_BLOCK_SIZE_2048>(
            num_blocks, y, _info.atype,
            stride_y_batch, stride_y_channel,
            x, stride_x_batch, stride_x_channel, stride_x_length,
            channels, input_length, output_length, ndim,
            musa_stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::adaptive_max_pool1d::moore
