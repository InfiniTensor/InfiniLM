#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "vander_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::vander::nvidia {

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const VanderInfo &info,
    void *stream) {

    // 1. 准备指针
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 2. 准备参数
    size_t rows = info.rows();
    size_t cols = info.cols();
    bool increasing = info.increasing();

    // 计算总元素数量以确定 Grid Size
    size_t total_elements = rows * cols;
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    // 调用 CUDA Kernel
    op::vander::cuda::vander_kernel<T>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr, in_ptr, rows, cols, increasing);
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
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    int N,
    int increasing) {

    // 1. 创建并校验 Info
    auto info_result = VanderInfo::create(out_desc, input_desc, N, increasing);
    if (!info_result) {
        return info_result.status();
    }

    // 2. 创建 Descriptor
    // Vander 算子是 Element-wise 展开操作，不需要额外的 workspace
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque(),
        info_result.take(),
        workspace_size,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    // 3. 根据数据类型分发 Kernel
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

} // namespace op::vander::nvidia
