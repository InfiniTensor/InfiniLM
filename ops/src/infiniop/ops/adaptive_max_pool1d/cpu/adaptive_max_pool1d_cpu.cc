#include "adaptive_max_pool1d_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include <algorithm>
#include <cmath>

namespace op::adaptive_max_pool1d::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t output_size) {
    auto result = AdaptiveMaxPool1dInfo::create(y_desc, x_desc, output_size);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t adaptiveMaxPool1d(const AdaptiveMaxPool1dInfo *info, T *y, const T *x) {

    const size_t ndim = info->ndim();
    const size_t batch_size = info->shape[0];
    const size_t channels = ndim > 2 ? info->shape[1] : 1;

    const size_t input_length = info->input_length();
    const size_t output_length = info->output_length();

    // 计算总的任务块数 (Batch * Channels)
    const ptrdiff_t total_blocks = static_cast<ptrdiff_t>(batch_size * channels);

    const ptrdiff_t x_stride_last = info->x_strides.back();

#pragma omp parallel for
    for (ptrdiff_t block_idx = 0; block_idx < total_blocks; ++block_idx) {
        const size_t i = block_idx / channels; // batch index
        const size_t j = block_idx % channels; // channel index

        const T *x_ptr_base;
        T *y_ptr_base;

        if (ndim > 2) { // (N, C, L)
            x_ptr_base = x + i * info->x_strides[0] + j * info->x_strides[1];
            y_ptr_base = y + i * info->y_strides[0] + j * info->y_strides[1];
        } else { // (N, L)
            x_ptr_base = x + i * info->x_strides[0];
            y_ptr_base = y + i * info->y_strides[0];
        }

        for (size_t out_idx = 0; out_idx < output_length; ++out_idx) {
            size_t start_index = (out_idx * input_length) / output_length;
            size_t end_index = ((out_idx + 1) * input_length + output_length - 1) / output_length;

            start_index = std::max(start_index, size_t(0));
            end_index = std::min(end_index, input_length);
            size_t window_len = end_index - start_index;

            if (window_len <= 0) {
                continue;
            }

            const T *window_ptr = x_ptr_base + start_index * x_stride_last;

            auto max_val = op::common_cpu::reduce_op::max(window_ptr, window_len, x_stride_last);
            y_ptr_base[out_idx] = utils::cast<T>(max_val);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x,
    void *stream) const {

    if (_info.atype == INFINI_DTYPE_F32) {
        return adaptiveMaxPool1d(&_info, (float *)y, (const float *)x);
    } else if (_info.atype == INFINI_DTYPE_F16) {
        return adaptiveMaxPool1d(&_info, (fp16_t *)y, (const fp16_t *)x);
    } else if (_info.atype == INFINI_DTYPE_BF16) {
        return adaptiveMaxPool1d(&_info, (bf16_t *)y, (const bf16_t *)x);
    } else if (_info.atype == INFINI_DTYPE_F64) {
        return adaptiveMaxPool1d(&_info, (double *)y, (const double *)x);
    }

    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}

} // namespace op::adaptive_max_pool1d::cpu
