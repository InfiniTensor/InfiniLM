#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "affine_grid_nvidia.cuh"

namespace op::affine_grid::nvidia {

// ==================================================================
// Kernel Launch Helper
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *input, // 这里对应 Theta
    size_t batch,
    size_t height,
    size_t width,
    bool align_corners,
    void *stream) {

    // 指针强转
    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);

    // 计算总线程数: N * H * W
    // 每个线程负责生成一个 (x, y) 坐标对
    size_t total_elements = batch * height * width;

    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    cuda::affine_grid_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(
        out_ptr,
        in_ptr,
        batch,
        height,
        width,
        align_corners);
}

// ==================================================================
// Descriptor Implementation
// ==================================================================

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

// 创建算子描述符
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t in_desc,
    bool align_corners) {

    // 1. 使用 Info 类解析并校验参数
    // create 方法内部会检查 Tensor 维度和类型一致性
    auto info_result = AffineGridInfo::create(out_desc, in_desc, align_corners);
    if (!info_result) {
        return info_result.status();
    }
    auto info = info_result.take();

    // 2. 创建 Descriptor
    *desc_ptr = new Descriptor(
        new Opaque(),     // Opaque 指针
        info,             // Info 对象 (包含 N, H, W, align_corners)
        0,                // Workspace size (AffineGrid 不需要额外 workspace)
        handle->device,   // Device Type
        handle->device_id // Device ID
    );

    return INFINI_STATUS_SUCCESS;
}

// 执行计算
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {
    auto dtype = _info.dtype();
    auto batch = _info.batch();
    auto height = _info.height();
    auto width = _info.width();
    auto align_corners = _info.align_corners();

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, batch, height, width, align_corners, stream);
        break;

    case INFINI_DTYPE_BF16:

        launch_kernel<cuda_bfloat16>(output, input, batch, height, width, align_corners, stream);
        break;

    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, batch, height, width, align_corners, stream);
        break;

    case INFINI_DTYPE_F64:

        launch_kernel<double>(output, input, batch, height, width, align_corners, stream);
        break;

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::affine_grid::nvidia
