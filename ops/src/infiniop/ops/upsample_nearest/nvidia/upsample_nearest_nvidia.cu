#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../handle.h"

#include "../cuda/kernel.cuh"
#include "upsample_nearest_nvidia.cuh"
#include <algorithm>
#include <cstdint>

namespace op::upsample_nearest::nvidia {

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
    const UpsampleNearestInfo &info,
    void *stream) {

    // 1. Prepare Pointers
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);

    // 2. Prepare Dimensions
    size_t N = info.n();
    size_t C = info.c();
    size_t H_in = info.h_in();
    size_t W_in = info.w_in();
    size_t H_out = info.h_out();
    size_t W_out = info.w_out();

    // 3. Pre-compute Scaling Factors on Host
    // Nearest neighbor scaling: in_size / out_size
    float scale_h = static_cast<float>(H_in) / H_out;
    float scale_w = static_cast<float>(W_in) / W_out;

    // 4. Configure Grid/Block
    // Total number of output elements
    size_t total_elements = N * C * H_out * W_out;
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    // Cap grid size to avoid launch failures on huge tensors
    if (grid_size > 65535) {
        grid_size = 65535;
    }

    op::upsample_nearest::cuda::upsample_nearest_kernel<T>
        <<<grid_size, block_size, 0, cuda_stream>>>(
            out_ptr,
            in_ptr,
            N, C, H_in, W_in, H_out, W_out,
            scale_h, scale_w);
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
    infiniopTensorDescriptor_t input_desc) {

    auto info_result = UpsampleNearestInfo::create(out_desc, input_desc);
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

    // Verify pointers
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
    // Nearest Neighbor 插值通常也支持整型 (如 Mask 处理)
    case INFINI_DTYPE_U8:
        launch_kernel<uint8_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_I8:
        launch_kernel<int8_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_I16:
        launch_kernel<int16_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_U16:
        launch_kernel<uint16_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_I32:
        launch_kernel<int32_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_U32:
        launch_kernel<uint32_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_I64:
        launch_kernel<int64_t>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_U64:
        launch_kernel<uint64_t>(output, input, _info, stream);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::upsample_nearest::nvidia
