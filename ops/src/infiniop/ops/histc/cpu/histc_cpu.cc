#include "histc_cpu.h"
#include "../../../../utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace op::histc::cpu {

utils::Result<HistcInfo> HistcInfo::create(
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    int64_t bins,
    double min_val,
    double max_val) {

    if (bins <= 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (min_val >= max_val) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto x_shape = x_desc->shape();
    auto y_shape = y_desc->shape();

    // Input should be 1D
    if (x_shape.size() != 1) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Output should be 1D with bins elements
    if (y_shape.size() != 1 || y_shape[0] != static_cast<size_t>(bins)) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    HistcInfo info;
    info.input_size = x_shape[0];
    info.bins = bins;
    info.min_val = min_val;
    info.max_val = max_val;
    info.input_stride = x_desc->strides()[0];
    info.output_stride = y_desc->strides()[0];

    // This implementation assumes x points to the first logical element and uses linear indexing.
    // Negative (or broadcasted) strides would require an explicit base offset.
    if (info.input_stride <= 0) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    // Writing a histogram into a broadcasted or negatively strided output is undefined.
    if (info.output_stride <= 0) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    return utils::Result<HistcInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int64_t bins,
    double min_val,
    double max_val) {

    auto dtype = x_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // Histc output is always float32.
    if (y_desc->dtype() != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    auto info_result = HistcInfo::create(x_desc, y_desc, bins, min_val, max_val);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void histc_impl(
    const HistcInfo &info,
    float *y,
    const T *x) {

    // Initialize output to zero (supports non-unit stride).
    for (int64_t b = 0; b < info.bins; ++b) {
        y[b * info.output_stride] = 0.0f;
    }

    const double bin_width = (info.max_val - info.min_val) / static_cast<double>(info.bins);

    for (size_t i = 0; i < info.input_size; ++i) {
        double val = utils::cast<double>(x[i * info.input_stride]);

        // Skip values outside range
        if (val < info.min_val || val > info.max_val) {
            continue;
        }

        // Calculate bin index
        int64_t bin_idx = static_cast<int64_t>((val - info.min_val) / bin_width);

        // Handle edge case: max_val should go to last bin
        if (bin_idx >= info.bins) {
            bin_idx = info.bins - 1;
        }
        if (bin_idx < 0) {
            bin_idx = 0;
        }

        y[bin_idx * info.output_stride] += 1.0f;
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    float *y_ptr = reinterpret_cast<float *>(y);

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        histc_impl<fp16_t>(_info, y_ptr, reinterpret_cast<const fp16_t *>(x));
        break;
    case INFINI_DTYPE_BF16:
        histc_impl<bf16_t>(_info, y_ptr, reinterpret_cast<const bf16_t *>(x));
        break;
    case INFINI_DTYPE_F32:
        histc_impl<float>(_info, y_ptr, reinterpret_cast<const float *>(x));
        break;
    case INFINI_DTYPE_F64:
        histc_impl<double>(_info, y_ptr, reinterpret_cast<const double *>(x));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::histc::cpu
