#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "triplet_margin_with_distance_loss_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::triplet_margin_with_distance_loss::nvidia {

struct Descriptor::Opaque {
    size_t batch_size;
    size_t feature_dim;
};

template <typename T>
void launch_kernel(
    void *output,
    void *workspace, // Workspace pointer (float*)
    const void *anchor,
    const void *positive,
    const void *negative,
    const TripletMarginWithDistanceLossInfo &info,
    size_t batch_size,
    size_t feature_dim,
    void *stream) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto ws_ptr = reinterpret_cast<float *>(workspace); // FP32 Workspace
    auto anchor_ptr = reinterpret_cast<const T *>(anchor);
    auto pos_ptr = reinterpret_cast<const T *>(positive);
    auto neg_ptr = reinterpret_cast<const T *>(negative);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    float margin = info.margin();
    int swap = info.swap();
    int reduction = info.reduction(); // 0:None, 1:Mean, 2:Sum

    size_t grid_size = batch_size;

    unsigned int threads_per_block = 256;
    if (feature_dim < 256) {
        threads_per_block = 128;
    }
    if (feature_dim < 128) {
        threads_per_block = 64;
    }
    if (feature_dim < 64) {
        threads_per_block = 32;
    }

    // 1. 初始化 Accumulator
    if (reduction != 0) {
        cudaMemsetAsync(ws_ptr, 0, sizeof(float), cuda_stream);
    }

    op::triplet_margin_with_distance_loss::cuda::triplet_margin_loss_kernel<T>
        <<<grid_size, threads_per_block, 0, cuda_stream>>>(
            out_ptr,
            ws_ptr, // 传递 workspace
            anchor_ptr,
            pos_ptr,
            neg_ptr,
            feature_dim,
            margin,
            swap,
            reduction,
            batch_size);

    // 3. 后处理: Cast & Mean
    if (reduction != 0) {
        op::triplet_margin_with_distance_loss::cuda::cast_and_scale_kernel<T>
            <<<1, 1, 0, cuda_stream>>>(
                out_ptr,
                ws_ptr,
                batch_size,
                reduction);
    }
}

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t anchor_desc,
    infiniopTensorDescriptor_t positive_desc,
    infiniopTensorDescriptor_t negative_desc,
    float margin,
    int swap,
    int reduction) {

    auto info_result = TripletMarginWithDistanceLossInfo::create(
        output_desc, anchor_desc, positive_desc, negative_desc, margin, swap, reduction);
    if (!info_result) {
        return info_result.status();
    }

    int ndim = anchor_desc->ndim();
    size_t feature_dim = (ndim > 0) ? anchor_desc->shape()[ndim - 1] : 1;
    size_t total_elements = info_result->num_elements();
    size_t batch_size = total_elements / feature_dim;

    auto opaque = new Opaque();
    opaque->batch_size = batch_size;
    opaque->feature_dim = feature_dim;
    size_t workspace_size = (reduction != 0) ? sizeof(float) : 0;

    *desc_ptr = new Descriptor(opaque, info_result.take(), workspace_size, handle->device, handle->device_id);
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
    size_t batch_size = _opaque->batch_size;
    size_t feature_dim = _opaque->feature_dim;

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, workspace, anchor, positive, negative, _info, batch_size, feature_dim, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, workspace, anchor, positive, negative, _info, batch_size, feature_dim, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, workspace, anchor, positive, negative, _info, batch_size, feature_dim, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, workspace, anchor, positive, negative, _info, batch_size, feature_dim, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::triplet_margin_with_distance_loss::nvidia
