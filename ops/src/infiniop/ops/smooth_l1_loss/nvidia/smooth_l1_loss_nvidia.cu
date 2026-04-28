#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "smooth_l1_loss_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::smooth_l1_loss::nvidia {

// ==================================================================
// 辅助函数
// ==================================================================
template <typename T>
bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output, const void *input, const void *target, void *workspace,
    size_t numel, float beta, int reduction,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto tar_ptr = reinterpret_cast<const T *>(target);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    op::smooth_l1_loss::cuda::SmoothL1LossFunctor functor(beta);

    // ------------------------------------------
    // 模式 1: Elementwise (None)
    // ------------------------------------------
    if (reduction == 0) {
        auto out_ptr = reinterpret_cast<T *>(output);
        constexpr int TotalBytes = 16;
        constexpr int PackSize = TotalBytes / sizeof(T);
        bool can_vectorize = (PackSize > 1) && (numel % PackSize == 0) && is_aligned<T>(output, TotalBytes) && is_aligned<T>(input, TotalBytes) && is_aligned<T>(target, TotalBytes);

        if (can_vectorize) {
            size_t num_packs = numel / PackSize;
            size_t block_size = 256;
            size_t grid_size = (num_packs + block_size - 1) / block_size;
            op::smooth_l1_loss::cuda::smooth_l1_loss_kernel_vectorized<T, PackSize>
                <<<grid_size, block_size, 0, cuda_stream>>>(out_ptr, in_ptr, tar_ptr, num_packs, functor);
        } else {
            size_t block_size = 256;
            size_t grid_size = (numel + block_size - 1) / block_size;
            op::smooth_l1_loss::cuda::smooth_l1_loss_kernel<T>
                <<<grid_size, block_size, 0, cuda_stream>>>(out_ptr, in_ptr, tar_ptr, numel, functor);
        }
    }
    // ------------------------------------------
    // 模式 2: Reduction (Mean / Sum)
    // ------------------------------------------
    else {
        // 使用 workspace 作为临时的 float 累加器 (精度更高，且方便 atomicAdd)
        float *acc_ptr = reinterpret_cast<float *>(workspace);

        // 1. 清零 Accumulator
        cudaMemsetAsync(acc_ptr, 0, sizeof(float), cuda_stream);

        // 2. 启动 Reduction Kernel
        float scale = (reduction == 1) ? (1.0f / numel) : 1.0f; // 1=Mean, 2=Sum
        size_t block_size = 256;
        // 限制 Grid 大小，避免过多 Block 竞争 atomicAdd
        size_t grid_size = std::min((numel + block_size - 1) / block_size, static_cast<size_t>(1024));

        op::smooth_l1_loss::cuda::smooth_l1_loss_reduce_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                acc_ptr, in_ptr, tar_ptr, numel, functor, scale);

        // 3. 将结果从 float workspace 转回目标类型并写入 output
        // 输出只有 1 个元素
        op::smooth_l1_loss::cuda::cast_float_to_t<T>
            <<<1, 1, 0, cuda_stream>>>(reinterpret_cast<T *>(output), acc_ptr);
    }
}

// ==================================================================
// Descriptor
// ==================================================================
struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t target_desc,
    float beta, int reduction) {

    auto info_result = SmoothL1LossInfo::create(out_desc, input_desc, target_desc, beta, reduction);
    if (!info_result) {
        return info_result.status();
    }

    // [关键] 如果是 Reduction 模式，我们需要 4 字节的 workspace 来存 float 中间结果
    size_t workspace_size = 0;
    if (reduction != 0) {
        workspace_size = sizeof(float);
    }

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size, void *output,
    const void *input, const void *target, void *stream) const {

    auto dtype = _info.dtype();
    auto numel = _info.numel();
    float beta = _info.beta();
    int reduction = _info.reduction();

    // 检查 workspace 是否够用
    if (reduction != 0 && workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, target, workspace, numel, beta, reduction, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, input, target, workspace, numel, beta, reduction, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, target, workspace, numel, beta, reduction, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, target, workspace, numel, beta, reduction, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::smooth_l1_loss::nvidia
