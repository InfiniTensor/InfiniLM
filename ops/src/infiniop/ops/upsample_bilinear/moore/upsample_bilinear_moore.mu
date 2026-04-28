#include "../../../devices/moore/moore_handle.h"
#include "upsample_bilinear_moore.h"
#include "upsample_bilinear_moore_kernel.h"
#include <algorithm>
#include <cstdint>
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>

namespace op::upsample_bilinear::moore {

template <typename T>
static inline bool is_aligned(const void *ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

template <typename T>
void launch_kernel(
    void *output,
    const void *input,
    const UpsampleBilinearInfo &info,
    void *stream) {

    auto in_ptr = reinterpret_cast<const T *>(input);
    auto out_ptr = reinterpret_cast<T *>(output);

    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

    size_t N = info.n();
    size_t C = info.c();
    size_t H_in = info.h_in();
    size_t W_in = info.w_in();
    size_t H_out = info.h_out();
    size_t W_out = info.w_out();
    bool align_corners = info.align_corners();

    float scale_h, scale_w;
    if (align_corners) {
        scale_h = (H_out > 1) ? static_cast<float>(H_in - 1) / (H_out - 1) : 0.0f;
        scale_w = (W_out > 1) ? static_cast<float>(W_in - 1) / (W_out - 1) : 0.0f;
    } else {
        scale_h = static_cast<float>(H_in) / H_out;
        scale_w = static_cast<float>(W_in) / W_out;
    }

    size_t total_elements = N * C * H_out * W_out;
    size_t block_size = 256;
    size_t grid_size = (total_elements + block_size - 1) / block_size;

    if (grid_size > 65535) {
        grid_size = 65535;
    }

    op::upsample_bilinear::moore::upsample_bilinear_kernel<T>
        <<<grid_size, block_size, 0, musa_stream>>>(
            out_ptr,
            in_ptr,
            N, C, H_in, W_in, H_out, W_out,
            scale_h, scale_w,
            align_corners);
}

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

    if (!output || !input) {
        return INFINI_STATUS_BAD_PARAM;
    }

    switch (dtype) {
    case INFINI_DTYPE_F16:
        launch_kernel<half>(output, input, _info, stream);
        break;
    case INFINI_DTYPE_BF16:
        launch_kernel<__mt_bfloat16>(output, input, _info, stream);
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

} // namespace op::upsample_bilinear::moore
