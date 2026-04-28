#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "adaptive_avg_pool3d_nvidia.cuh"
#include <cstddef>

template <typename T, unsigned int BLOCK_SIZE>
INFINIOP_CUDA_KERNEL
adaptiveAvgPool3D(T *y, const T *x, size_t N, size_t C, size_t x_d, size_t x_h,
                  size_t x_w, size_t y_d, size_t y_h, size_t y_w,
                  const ptrdiff_t *x_strides, const ptrdiff_t *y_strides) {
    adaptiveAvgPool3DKernel<T, BLOCK_SIZE>(y, x, N, C, x_d, x_h, x_w, y_d, y_h,
                                           y_w, x_strides, y_strides);
}

namespace op::adaptive_avg_pool3d::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t y_desc,
                                  infiniopTensorDescriptor_t x_desc,
                                  size_t *output_size) {
    auto info = AdaptiveAvgPool3DInfo::create(y_desc, x_desc, output_size);
    CHECK_RESULT(info);
    size_t workspace_size = 10 * sizeof(ptrdiff_t); // for x_strides and y_strides
    *desc_ptr = new Descriptor(
        info.take(), workspace_size,
        new Opaque{
            reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t dtype,
                            size_t N, size_t C, size_t x_d, size_t x_h,
                            size_t x_w, size_t y_d, size_t y_h, size_t y_w,
                            const ptrdiff_t *x_strides,
                            const ptrdiff_t *y_strides, cudaStream_t stream) {
    size_t num_blocks = (N * C * y_d * y_h * y_w + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (dtype == INFINI_DTYPE_F16) {
        adaptiveAvgPool3D<half, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (half *)y, (const half *)x, N, C, x_d, x_h, x_w, y_d, y_h, y_w,
            x_strides, y_strides);
    } else if (dtype == INFINI_DTYPE_F32) {
        adaptiveAvgPool3D<float, BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            (float *)y, (const float *)x, N, C, x_d, x_h, x_w, y_d, y_h, y_w,
            x_strides, y_strides);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y, const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;

    // Prepare strides
    ptrdiff_t *x_strides = (ptrdiff_t *)workspace;
    ptrdiff_t *y_strides = (ptrdiff_t *)workspace + _info.x_strides.size();
    cudaMemcpyAsync(x_strides, _info.x_strides.data(),
                    _info.x_strides.size() * sizeof(ptrdiff_t),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(y_strides, _info.y_strides.data(),
                    _info.y_strides.size() * sizeof(ptrdiff_t),
                    cudaMemcpyHostToDevice, stream);

    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            y, x, _info.dtype, _info.N, _info.C, _info.x_d, _info.x_h, _info.x_w,
            _info.y_d, _info.y_h, _info.y_w, x_strides, y_strides, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            y, x, _info.dtype, _info.N, _info.C, _info.x_d, _info.x_h, _info.x_w,
            _info.y_d, _info.y_h, _info.y_w, x_strides, y_strides, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_2048>(
            y, x, _info.dtype, _info.N, _info.C, _info.x_d, _info.x_h, _info.x_w,
            _info.y_d, _info.y_h, _info.y_w, x_strides, y_strides, stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            y, x, _info.dtype, _info.N, _info.C, _info.x_d, _info.x_h, _info.x_w,
            _info.y_d, _info.y_h, _info.y_w, x_strides, y_strides, stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::adaptive_avg_pool3d::nvidia
