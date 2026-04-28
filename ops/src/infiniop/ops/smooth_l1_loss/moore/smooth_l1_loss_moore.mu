#include "smooth_l1_loss_moore.h"
#include "smooth_l1_loss_moore_kernel.h"

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

#include "../../../devices/moore/moore_handle.h"

namespace op::smooth_l1_loss::moore {

template <typename T>
void smooth_l1_loss_moore_launch(
    const SmoothL1LossInfo &info,
    T *output,
    const T *input,
    const T *target,
    void *stream) {

    auto musa_stream = (musaStream_t)stream;
    size_t numel = info.numel();
    int reduction = info.reduction(); // 0:None, 1:Mean, 2:Sum

    int threads = 256;
    // Calculate blocks
    int blocks = (numel + threads - 1) / threads;
    // Cap blocks to avoid overhead for huge tensors (standard practice)
    if (blocks > 1024) {
        blocks = 1024;
    }

    if (reduction == 0) {
        // --- None (Elementwise) ---
        // Just use simple grid mapping
        int simple_blocks = (numel + threads - 1) / threads;
        smooth_l1_loss_elementwise_kernel<T><<<simple_blocks, threads, 0, musa_stream>>>(
            numel, info.beta(), input, target, output);
    } else {
        // --- Mean / Sum (Reduction) ---

        // 1. Zero out output (Atomic accumulation destination)
        musaMemsetAsync(output, 0, sizeof(T), musa_stream);

        // 2. Launch Reduction Kernel
        // Shared Memory size: threads * sizeof(float) because we accumulate in float32
        size_t smem_size = threads * sizeof(float);

        smooth_l1_loss_reduce_kernel<T><<<blocks, threads, smem_size, musa_stream>>>(
            numel, info.beta(), input, target, output);

        // 3. Post-processing for Mean
        if (reduction == 1) {
            avg_scaling_kernel<T><<<1, 1, 0, musa_stream>>>(output, numel);
        }
    }
}

// ... (Descriptor implementation remains unchanged) ...
// Ensure Descriptor::~Descriptor(), create, calculate are still there as before.
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    float beta,
    int reduction) {

    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto info_result = SmoothL1LossInfo::create(out_desc, input_desc, target_desc, beta, reduction);
    if (!info_result) {
        return info_result.status();
    }

    *desc_ptr = new Descriptor(nullptr, *info_result, 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *target,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    switch (_info.dtype()) {
    case INFINI_DTYPE_F16:
        smooth_l1_loss_moore_launch<half>(_info, static_cast<half *>(output), static_cast<const half *>(input), static_cast<const half *>(target), stream);
        break;
    case INFINI_DTYPE_BF16:
        smooth_l1_loss_moore_launch<__mt_bfloat16>(_info, static_cast<__mt_bfloat16 *>(output), static_cast<const __mt_bfloat16 *>(input), static_cast<const __mt_bfloat16 *>(target), stream);
        break;
    case INFINI_DTYPE_F32:
        smooth_l1_loss_moore_launch<float>(_info, static_cast<float *>(output), static_cast<const float *>(input), static_cast<const float *>(target), stream);
        break;
    case INFINI_DTYPE_F64:
        smooth_l1_loss_moore_launch<double>(_info, static_cast<double *>(output), static_cast<const double *>(input), static_cast<const double *>(target), stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::smooth_l1_loss::moore
