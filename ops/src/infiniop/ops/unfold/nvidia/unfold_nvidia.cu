#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "unfold_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::unfold::nvidia {

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const UnfoldInfo &info,
    void *stream) {

    // 1. 准备指针
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 2. 准备参数 (从 Info 的向量中解包)
    // 注意：目前的 CUDA Kernel 仅支持 2D Spatial (NCHW)，这里取前两个维度
    if (info._kernel_sizes.size() < 2) {
        // 异常情况处理，或者直接返回
        return;
    }

    int C = info._C_in;
    int H = info._input_spatial_shape[0];
    int W = info._input_spatial_shape[1];

    int out_h = info._output_spatial_shape[0];
    int out_w = info._output_spatial_shape[1];

    int k_h = info._kernel_sizes[0];
    int k_w = info._kernel_sizes[1];
    int pad_h = info._paddings[0];
    int pad_w = info._paddings[1];
    int stride_h = info._strides[0];
    int stride_w = info._strides[1];
    int dil_h = info._dilations[0];
    int dil_w = info._dilations[1];

    // 3. 计算 Grid
    // 输出通道数 = C * kH * kW
    size_t out_channels = info._C_out;
    size_t out_spatial = info._L;
    size_t total_elements = info._N * out_channels * out_spatial;

    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    // 4. 调用 CUDA Kernel
    op::unfold::cuda::unfold_kernel<T>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr, in_ptr,
            C, H, W,
            out_h, out_w,
            k_h, k_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dil_h, dil_w,
            total_elements);
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
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    const int *kernel_sizes,
    const int *strides,
    const int *paddings,
    const int *dilations) {

    auto handle = reinterpret_cast<const infiniopHandle_t>(handle_);

    // 1. 创建并校验 Info
    // 使用新的 infer 接口
    auto result = UnfoldInfo::infer(out_desc, input_desc, kernel_sizes, strides, paddings, dilations);
    if (!result) {
        return result.status();
    }

    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
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

    // 使用新的 getter
    auto dtype = _info.dtype_val();

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

} // namespace op::unfold::nvidia
