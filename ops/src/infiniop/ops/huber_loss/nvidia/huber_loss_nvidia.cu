#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "huber_loss_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::huber_loss::nvidia {
template <typename T>
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const void *target,
    void *workspace,
    const HuberLossInfo &info,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);
    auto tar_ptr = reinterpret_cast<const T *>(target);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    size_t count = info.count();
    int reduction = info.reduction();

    op::huber_loss::cuda::HuberLossFunctor functor(info.delta());

    if (reduction == 0) {
        size_t block_size = 256;
        size_t grid_size = (count + block_size - 1) / block_size;

        op::huber_loss::cuda::huber_loss_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                out_ptr, in_ptr, tar_ptr, count, functor);
    } else {
        float *acc_ptr = reinterpret_cast<float *>(workspace);
        cudaMemsetAsync(acc_ptr, 0, sizeof(float), cuda_stream);
        float scale = (reduction == 1) ? (1.0f / static_cast<float>(count)) : 1.0f;

        size_t block_size = 256;
        size_t grid_size = std::min((count + block_size - 1) / block_size, static_cast<size_t>(1024));

        op::huber_loss::cuda::huber_loss_reduce_kernel<T>
            <<<grid_size, block_size, 0, cuda_stream>>>(
                acc_ptr, in_ptr, tar_ptr, count, functor, scale);
        op::huber_loss::cuda::cast_float_to_t<T>
            <<<1, 1, 0, cuda_stream>>>(out_ptr, acc_ptr);
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
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    float delta,
    int reduction) {

    auto info_result = HuberLossInfo::create(out_desc, input_desc, target_desc, delta, reduction);
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
    const void *input,
    const void *target,
    void *stream) const {

    auto dtype = _info.dtype();
    int reduction = _info.reduction();

    if (reduction != 0 && workspace_size < sizeof(float)) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, target, workspace, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<cuda_bfloat16>(output, input, target, workspace, _info, stream);
        break;
    case INFINI_DTYPE_F32:
        launch_kernel<float>(output, input, target, workspace, _info, stream);
        break;
    case INFINI_DTYPE_F64:
        launch_kernel<double>(output, input, target, workspace, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::huber_loss::nvidia
