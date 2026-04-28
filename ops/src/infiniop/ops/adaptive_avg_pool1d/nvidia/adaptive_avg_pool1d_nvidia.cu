
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "adaptive_avg_pool1d_nvidia.cuh"

namespace op::adaptive_avg_pool1d::nvidia {

template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    size_t num_channels, // 这里实际上是 total_channels (Batch * C)
    size_t isize,
    size_t osize,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    cuda::launch_adaptive_avg_pool1d<T>(
        out_ptr,
        in_ptr,
        num_channels,
        isize,
        osize,
        cuda_stream);
}

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
    infiniopTensorDescriptor_t in_desc) {

    // 1. 使用 Info 类解析参数
    auto info_result = AdaptiveAvgPool1dInfo::create(out_desc, in_desc);
    if (!info_result) {
        return info_result.status();
    }
    auto info = info_result.take();

    // 2. 创建 Descriptor
    *desc_ptr = new Descriptor(
        new Opaque(),     // Opaque 指针
        info,             // Info 对象
        0,                // Workspace size
        handle->device,   // Device Type
        handle->device_id // Device ID
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();
    auto num_channels = _info.num_channels(); // 这里通常是 Batch * Channel
    auto input_size = _info.input_size();
    auto output_size = _info.output_size();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, num_channels, input_size, output_size, stream);
        break;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, input, num_channels, input_size, output_size, stream);
        break;
#endif
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, num_channels, input_size, output_size, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, num_channels, input_size, output_size, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::adaptive_avg_pool1d::nvidia
