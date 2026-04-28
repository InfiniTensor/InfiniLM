#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh" // 假设这里包含了一些通用的 CUDA 宏或工具
#include "log_softmax_nvidia.cuh"

#include <algorithm>
#include <cstdint>

namespace op::log_softmax::nvidia {

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const LogSoftmaxInfo &info,
    void *stream) {

    // 1. 准备指针
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 2. 准备形状参数
    size_t dim_size = info.dim_size();
    size_t outer_size = info.outer_size();
    size_t inner_size = info.inner_size();

    // 3. 计算 Grid/Block
    // Grid: 总切片数 (Outer * Inner)
    // 每个 Block 处理 1 个 Slice (Dim 维度)
    size_t total_slices = outer_size * inner_size;

    // Block: 选择一个合理的 Block Size (例如 256)
    // Kernel 内部使用了循环处理 dim_size > blockDim 的情况，
    // 同时使用了 warp reduce，建议 blockDim 至少为 32。
    unsigned int threads_per_block = 256;

    // 如果 dim_size 很小，可以适当减小 block size，但不要小于 32 (Warp Size)
    if (dim_size < 256) {
        threads_per_block = 128;
    }
    if (dim_size < 128) {
        threads_per_block = 64;
    }
    if (dim_size < 64) {
        threads_per_block = 32;
    }

    // 4. 启动 Kernel
    // Shared memory 在 kernel 内部静态分配，此处不需要动态分配
    op::log_softmax::cuda::log_softmax_kernel<T>
        <<<total_slices, threads_per_block, 0, cuda_stream>>>(
            out_ptr,
            in_ptr,
            dim_size,
            inner_size);
}

// ==================================================================
// Descriptor 实现
// ==================================================================
struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int dim) {

    auto info_result = LogSoftmaxInfo::create(output_desc, input_desc, dim);
    if (!info_result) {
        return info_result.status();
    }
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::log_softmax::nvidia
