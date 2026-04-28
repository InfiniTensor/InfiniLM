#include "../../../devices/moore/moore_handle.h"
#include "multi_margin_loss_moore.h"
#include "multi_margin_loss_moore_kernel.h"
#include <algorithm>
#include <cstdint>

namespace op::multi_margin_loss::moore {

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
    const void *input,
    const void *target,
    const void *weight,
    void *workspace,
    const MultiMarginLossInfo &info,
    void *stream) {

    // 1. 准备指针
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);
    // Target 在 Info 校验阶段已确保为 Int64
    auto tar_ptr = reinterpret_cast<const int64_t *>(target);
    // Weight 是可选的
    auto w_ptr = (weight != nullptr) ? reinterpret_cast<const T *>(weight) : nullptr;

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    // 2. 准备参数
    size_t N = info.batch_size();
    size_t C = info.num_classes();
    int reduction = info.reduction();

    op::multi_margin_loss::moore::MultiMarginLossFunctor functor(info.p(), info.margin());

    // ------------------------------------------
    // 模式 1: Elementwise (Reduction = None)
    // ------------------------------------------
    if (reduction == 0) {
        // 每个线程处理一个样本 N
        size_t block_size = 256;
        size_t grid_size = (N + block_size - 1) / block_size;

        op::multi_margin_loss::moore::multi_margin_loss_kernel<T>
            <<<grid_size, block_size, 0, musa_stream>>>(
                out_ptr, in_ptr, tar_ptr, w_ptr, N, C, functor);
    }
    // ------------------------------------------
    // 模式 2: Reduction (Mean / Sum)
    // ------------------------------------------
    else {
        // 使用 workspace 作为临时的 float 累加器 (精度更高，且方便 atomicAdd)
        float *acc_ptr = reinterpret_cast<float *>(workspace);
        musaMemsetAsync(acc_ptr, 0, sizeof(float), musa_stream);
        float scale = (reduction == 1) ? (1.0f / static_cast<float>(N)) : 1.0f; // 1=Mean, 2=Sum

        size_t block_size = 256;
        size_t grid_size = std::min((N + block_size - 1) / block_size, static_cast<size_t>(1024));

        op::multi_margin_loss::moore::multi_margin_loss_reduce_kernel<T>
            <<<grid_size, block_size, 0, musa_stream>>>(
                acc_ptr, in_ptr, tar_ptr, w_ptr, N, C, functor, scale);

        // 将 float 累加结果转回 T 写入 output
        op::multi_margin_loss::moore::cast_float_to_t<T>
            <<<1, 1, 0, musa_stream>>>(out_ptr, acc_ptr);
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
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t weight_desc,
    int p,
    float margin,
    int reduction) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);

    auto info_result = MultiMarginLossInfo::create(out_desc, input_desc, target_desc, weight_desc, p, margin, reduction);
    if (!info_result) {
        return info_result.status();
    }

    size_t workspace_size = 0;
    if (reduction != 0) {
        workspace_size = sizeof(float);
    }

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
    const void *target,
    const void *weight,
    void *stream) const {

    auto dtype = _info.dtype();
    int reduction = _info.reduction();

    // 检查 workspace 是否够用
    if (reduction != 0 && workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, target, weight, workspace, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__mt_bfloat16>(output, input, target, weight, workspace, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, target, weight, workspace, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, target, weight, workspace, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::multi_margin_loss::moore
