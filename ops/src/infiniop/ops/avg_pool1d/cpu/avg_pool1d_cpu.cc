#include "avg_pool1d_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>

namespace op::avg_pool1d::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t kernel_size,
    size_t stride,
    size_t padding) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = AvgPool1dInfo::createAvgPool1dInfo(y_desc, x_desc, kernel_size, stride, padding);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t calculateAvgPool1d(const AvgPool1dInfo &info,
                                  T *y,
                                  const T *x) {
    const float inv_kernel = 1.0f / static_cast<float>(info.kernel_size);

#pragma omp parallel for
    for (ptrdiff_t bc = 0; bc < ptrdiff_t(info.batch * info.channels); ++bc) {

        ptrdiff_t b = bc / info.channels;
        ptrdiff_t c = bc % info.channels;

        size_t y_base = b * info.y_stride_batch + c * info.y_stride_channel;
        size_t x_base = b * info.x_stride_batch + c * info.x_stride_channel;

        for (size_t ow = 0; ow < info.out_width; ++ow) {
            size_t y_offset = y_base + ow * info.y_stride_width;

            long long start_w = static_cast<long long>(ow * info.stride) - info.padding;
            long long end_w = start_w + info.kernel_size;

            long long valid_start = std::max(0LL, start_w);
            long long valid_end = std::min(static_cast<long long>(info.in_width), end_w);

            float sum = 0.0f;
            for (long long iw = valid_start; iw < valid_end; ++iw) {
                size_t x_offset = x_base + iw * info.x_stride_width;
                sum += utils::cast<float>(x[x_offset]);
            }

            const float avg = sum * inv_kernel;
            y[y_offset] = utils::cast<T>(avg);
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE(TDATA) calculateAvgPool1d(_info, (TDATA *)y, (const TDATA *)x)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return CALCULATE(fp16_t);
    case INFINI_DTYPE_BF16:
        return CALCULATE(bf16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE(float);
    case INFINI_DTYPE_F64:
        return CALCULATE(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE

} // namespace op::avg_pool1d::cpu
