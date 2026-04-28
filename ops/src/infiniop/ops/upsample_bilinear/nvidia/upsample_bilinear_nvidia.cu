#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "upsample_bilinear_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::upsample_bilinear::nvidia {

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
    const UpsampleBilinearInfo &info,
    void *stream) {

    // 1. Prepare Pointers
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 2. Prepare Dimensions and Parameters
    // We treat the input as [Batch*Channel, 1, H_in, W_in] conceptually in the kernel logic
    // or just pass N, C, H, W.
    // The kernel expects N, C, H, W to calculate indexing.
    size_t N = info.n();
    size_t C = info.c();
    size_t H_in = info.h_in();
    size_t W_in = info.w_in();
    size_t H_out = info.h_out();
    size_t W_out = info.w_out();
    bool align_corners = info.align_corners();

    // 3. Pre-compute Scaling Factors on Host
    // This avoids division in the kernel for every pixel.
    float scale_h, scale_w;
    if (align_corners) {
        scale_h = (H_out > 1) ? static_cast<float>(H_in - 1) / (H_out - 1) : 0.0f;
        scale_w = (W_out > 1) ? static_cast<float>(W_in - 1) / (W_out - 1) : 0.0f;
    } else {
        scale_h = static_cast<float>(H_in) / H_out;
        scale_w = static_cast<float>(W_in) / W_out;
    }

    // 4. Configure Grid/Block
    // Total number of output elements
    size_t total_elements = N * C * H_out * W_out;
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    // Cap grid size to avoid launch failures on huge tensors
    // The kernel uses a grid-stride loop, so it handles arbitrary sizes.
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    op::upsample_bilinear::cuda::upsample_bilinear_kernel<T>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr,
            in_ptr,
            N, C, H_in, W_in, H_out, W_out,
            scale_h, scale_w,
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

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    int align_corners) {

    auto info_result = UpsampleBilinearInfo::create(out_desc, input_desc, align_corners);
    if (!info_result) {
        return info_result.status();
    }

    // No extra workspace needed for this op
    size_t workspace_size = 0;

    *desc_ptr = new Descriptor(new Opaque(), info_result.take(), workspace_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    auto dtype = _info.dtype();

    // Verify pointers (optional but good practice)
    if (!output || !input) {
        return INFINI_STATUS_BAD_PARAM;
    }

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

} // namespace op::upsample_bilinear::nvidia
