#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "triplet_margin_loss_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::triplet_margin_loss::nvidia {

template <typename T>
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

// ==================================================================
// Kernel Launch Logic
// ==================================================================
template <typename T>
void launch_kernel(
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative,
    void *workspace,
    const TripletMarginLossInfo &info,
    void *stream) {

    // 1. 准备指针
    auto out_ptr = reinterpret_cast<T *>(output);
    auto anc_ptr = reinterpret_cast<const T *>(anchor);
    auto pos_ptr = reinterpret_cast<const T *>(positive);
    auto neg_ptr = reinterpret_cast<const T *>(negative);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 2. 准备参数
    size_t N = info.batch_size();
    size_t D = info.feature_dim();
    int reduction = info.reduction();

    // 创建 Functor
    op::triplet_margin_loss::cuda::TripletMarginLossFunctor functor(
        info.margin(),
        info.p(),
        info.eps(),
        info.swap());

    // ------------------------------------------
    // 模式 1: Pointwise (Reduction = None)
    // ------------------------------------------
    if (reduction == 0) {
        // 每个线程处理一个样本 N
        size_t block_size = 256;
        size_t grid_size = (N + block_size - 1) / block_size;

        op::triplet_margin_loss::cuda::triplet_margin_loss_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                out_ptr, anc_ptr, pos_ptr, neg_ptr, N, D, functor);
    }
    // ------------------------------------------
    // 模式 2: Reduction (Mean / Sum)
    // ------------------------------------------
    else {
        // 使用 workspace 作为临时的 float 累加器 (精度更高，且方便 atomicAdd)
        float *acc_ptr = reinterpret_cast<float *>(workspace);
        cudaMemsetAsync(acc_ptr, 0, sizeof(float), cuda_stream);

        float scale = (reduction == 1) ? (1.0f / static_cast<float>(N)) : 1.0f; // 1=Mean, 2=Sum

        // Grid Stride Loop 配置
        size_t block_size = 256;
        size_t grid_size = std::min((N + block_size - 1) / block_size, static_cast<size_t>(1024));

        op::triplet_margin_loss::cuda::triplet_margin_loss_reduce_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                acc_ptr, anc_ptr, pos_ptr, neg_ptr, N, D, functor, scale);

        // 将结果从 float 转回 T 并写入 output
        op::triplet_margin_loss::cuda::cast_float_to_t<T>
            <<<1, 1, 0, cuda_stream>>>(out_ptr, acc_ptr);
    }
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
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t anchor_desc,
    infiniopTensorDescriptor_t positive_desc,
    infiniopTensorDescriptor_t negative_desc,
    float margin,
    int p,
    float eps,
    int swap,
    int reduction) {

    auto info_result = TripletMarginLossInfo::create(out_desc, anchor_desc, positive_desc, negative_desc, margin, p, eps, swap, reduction);
    if (!info_result) {
        return info_result.status();
    }

    // 如果需要 Reduction，分配一个 float 大小的 workspace 用于 accumulator
    size_t workspace_size = 0;
    if (reduction != 0) {
        workspace_size = sizeof(float);
    }

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative,
    void *stream) const {

    auto dtype = _info.dtype();
    int reduction = _info.reduction();

    // 检查 workspace 是否够用
    if (reduction != 0 && workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, anchor, positive, negative, workspace, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, anchor, positive, negative, workspace, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, anchor, positive, negative, workspace, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, anchor, positive, negative, workspace, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::triplet_margin_loss::nvidia
