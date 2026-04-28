#include "../../../devices/moore/moore_handle.h"
#include "triplet_margin_loss_moore.h"
#include "triplet_margin_loss_moore_kernel.h"
#include <algorithm>
#include <cstdint>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::triplet_margin_loss::moore {

template <typename T>
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename T>
void launch_kernel(
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative,
    void *workspace,
    const TripletMarginLossInfo &info,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto anc_ptr = reinterpret_cast<const T *>(anchor);
    auto pos_ptr = reinterpret_cast<const T *>(positive);
    auto neg_ptr = reinterpret_cast<const T *>(negative);

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t N = info.batch_size();
    size_t D = info.feature_dim();
    int reduction = info.reduction();

    op::triplet_margin_loss::moore::TripletMarginLossFunctor functor(
        info.margin(),
        info.p(),
        info.eps(),
        info.swap());

    if (reduction == 0) {
        size_t block_size = 256;
        size_t grid_size = (N + block_size - 1) / block_size;

        op::triplet_margin_loss::moore::triplet_margin_loss_kernel<T>
            <<<grid_size, block_size, 0, musa_stream>>>(
                out_ptr, anc_ptr, pos_ptr, neg_ptr, N, D, functor);
    } else {
        float *acc_ptr = reinterpret_cast<float *>(workspace);
        musaMemsetAsync(acc_ptr, 0, sizeof(float), musa_stream);

        float scale = (reduction == 1) ? (1.0f / static_cast<float>(N)) : 1.0f;

        size_t block_size = 256;
        size_t grid_size = std::min((N + block_size - 1) / block_size, static_cast<size_t>(1024));

        op::triplet_margin_loss::moore::triplet_margin_loss_reduce_kernel<T>
            <<<grid_size, block_size, 0, musa_stream>>>(
                acc_ptr, anc_ptr, pos_ptr, neg_ptr, N, D, functor, scale);

        op::triplet_margin_loss::moore::cast_float_to_t<T>
            <<<1, 1, 0, musa_stream>>>(out_ptr, acc_ptr);
    }
}

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

    if (reduction != 0 && workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, anchor, positive, negative, workspace, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__mt_bfloat16>(output, anchor, positive, negative, workspace, _info, stream);
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

} // namespace op::triplet_margin_loss::moore
