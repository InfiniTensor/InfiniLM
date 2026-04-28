#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "argwhere_nvidia.cuh"
#include "infinicore.h"
#include <cstddef>
#include <cstdint>

int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

infiniStatus_t launchKernel(const void *data, int64_t *tmp, int64_t *results,
                            int64_t *block_sum, size_t N, const size_t *shapes,
                            const ptrdiff_t *strides, size_t ndim,
                            infiniDtype_t dtype, size_t *count,
                            int maxThreadsPerBlock, cudaStream_t stream) {

    if (dtype == INFINI_DTYPE_F32) {
        int block_size = maxThreadsPerBlock * 2;
        int num_blocks = (N + block_size - 1) / block_size;
        parallel_large_argwhere_kernel<float>
            <<<num_blocks, maxThreadsPerBlock, sizeof(int64_t) * block_size>>>(
                static_cast<const float *>(data), block_sum, tmp, N, shapes,
                strides, ndim);
        if (num_blocks > 1) {
            // 计算前缀和
            parallel_block_scan_kernel<int64_t>
                <<<1, maxThreadsPerBlock, sizeof(int64_t) * block_size>>>(
                    num_blocks + 1, block_sum);
            // 重新整理结果
            add_block_offset_kernel<<<num_blocks, maxThreadsPerBlock>>>(
                results, tmp, block_sum, ndim);
            cudaMemcpyAsync(count, &block_sum[num_blocks], sizeof(size_t),
                            cudaMemcpyDeviceToHost, stream);
        } else {
            cudaMemcpyAsync(count, &block_sum[0], sizeof(size_t),
                            cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(results, tmp, sizeof(int64_t) * (*count) * ndim,
                            cudaMemcpyDeviceToDevice, stream);
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

namespace op::argwhere::nvidia {
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(infiniopHandle_t handle,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t x_desc) {
    auto info = ArgwhereInfo::create(x_desc);
    CHECK_RESULT(info);
    Opaque *opaque = new Opaque{
        reinterpret_cast<device::nvidia::Handle *>(handle)->internal()};
    size_t workspace_size = x_desc->ndim() * sizeof(size_t) * 2 + x_desc->ndim() * sizeof(int64_t) * x_desc->numel() * 2 + sizeof(int64_t) * opaque->internal->maxThreadsPerBlock();
    *desc_ptr = new Descriptor(info.take(), workspace_size, opaque,
                               handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void **y, size_t *count, const void *x,
                                     void *stream) const {
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    size_t ndim = _info.strides.size();
    ptrdiff_t *strides = static_cast<ptrdiff_t *>(workspace);
    size_t *shapes = reinterpret_cast<size_t *>(strides + ndim);
    int64_t *tmp = reinterpret_cast<int64_t *>(shapes + ndim);
    int64_t *result = reinterpret_cast<int64_t *>(tmp + _info.num_elements * ndim);
    int64_t *block_sum = reinterpret_cast<int64_t *>(result + _info.num_elements * ndim);

    cudaMemcpyAsync(shapes, _info.shapes.data(),
                    _info.shapes.size() * sizeof(size_t), cudaMemcpyHostToDevice,
                    cuda_stream);
    cudaMemcpyAsync(strides, _info.strides.data(),
                    _info.strides.size() * sizeof(ptrdiff_t),
                    cudaMemcpyHostToDevice, cuda_stream);

    CHECK_STATUS(launchKernel(x, tmp, result, block_sum, _info.num_elements,
                              shapes, strides, ndim, INFINI_DTYPE_F32, count,
                              _opaque->internal->maxThreadsPerBlock(),
                              cuda_stream));
    // 从设备内存中读取 count_cuda 的值

    cudaStreamSynchronize(cuda_stream);

    // 写回结果
    *y = new int64_t[(*count) * ndim];

    cudaMemcpyAsync(*y, result, sizeof(int64_t) * (*count) * ndim,
                    cudaMemcpyDeviceToHost, cuda_stream);

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::argwhere::nvidia
