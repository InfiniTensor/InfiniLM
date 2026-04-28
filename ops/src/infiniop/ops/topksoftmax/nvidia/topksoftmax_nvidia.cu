#include "../../../devices/nvidia/nvidia_common.cuh"
#include "topksoftmax_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

namespace op::topksoftmax::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc) {
    auto result = TopksoftmaxInfo::create(x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    if (info.x_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        std::move(info),
        0,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <int BLOCK_SIZE = 128>
infiniStatus_t launch_topksoftmax(float *d_values_out, int *d_indices_out, const void *d_input, const size_t N, const size_t width, const size_t topk, const bool norm, infiniDtype_t xtype, cudaStream_t stream) {
    const int block_threads = BLOCK_SIZE;
    dim3 blocks(static_cast<unsigned int>(N));
    dim3 threads(block_threads);

    if (xtype == INFINI_DTYPE_F32) {
        softmax_topk_row_kernel<float, BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_values_out, d_indices_out, (float *)d_input, N, width, topk, norm);
    } else if (xtype == INFINI_DTYPE_F16) {
        softmax_topk_row_kernel<half, BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_values_out, d_indices_out, (half *)d_input, N, width, topk, norm);
    } else if (xtype == INFINI_DTYPE_BF16) {
        softmax_topk_row_kernel<__nv_bfloat16, BLOCK_SIZE><<<blocks, threads, 0, stream>>>(d_values_out, d_indices_out, (__nv_bfloat16 *)d_input, N, width, topk, norm);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

}; // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    float *values,
    int *indices,
    const void *x,
    const size_t topk,
    const bool norm,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    size_t N = _info.N;
    size_t width = _info.width;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (width <= 128) {
        launch_topksoftmax<128>(values, indices, x, N, width, topk, norm, _info.xtype, cuda_stream);
    } else if (width <= 256) {
        launch_topksoftmax<256>(values, indices, x, N, width, topk, norm, _info.xtype, cuda_stream);
    } else if (width <= 512) {
        launch_topksoftmax<512>(values, indices, x, N, width, topk, norm, _info.xtype, cuda_stream);
    } else {
        return INFINI_STATUS_INTERNAL_ERROR;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::topksoftmax::nvidia
