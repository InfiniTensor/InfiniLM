#include "adaptive_avg_pool3d_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <array>
namespace op::adaptive_avg_pool3d::cpu {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t *output_size) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto info = AdaptiveAvgPool3DInfo::create(y_desc, x_desc, output_size);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateAdaptiveAvgPool3D(
    const AdaptiveAvgPool3DInfo &info,
    Tdata *y,
    const Tdata *x) {

    std::array<size_t, 5> y_strides;
    y_strides[4] = 1;
    y_strides[3] = info.y_w * y_strides[4];
    y_strides[2] = info.y_h * y_strides[3];
    y_strides[1] = info.y_d * y_strides[2];
    y_strides[0] = info.C * y_strides[1];

    const size_t total = info.N * info.C * info.y_d * info.y_h * info.y_w;

#pragma omp parallel for
    for (ptrdiff_t idx = 0; idx < (ptrdiff_t)total; idx++) {

        // 🔹 Decode flattened index → (n, c, od, oh, ow)
        ptrdiff_t tmp = idx;

        ptrdiff_t ow = tmp % info.y_w;
        tmp /= info.y_w;
        ptrdiff_t oh = tmp % info.y_h;
        tmp /= info.y_h;
        ptrdiff_t od = tmp % info.y_d;
        tmp /= info.y_d;
        ptrdiff_t c = tmp % info.C;
        tmp /= info.C;
        ptrdiff_t n = tmp;

        // 🔹 Compute pooling region
        size_t x_start_d = od * info.x_d / info.y_d;
        size_t x_end_d = ((od + 1) * info.x_d + info.y_d - 1) / info.y_d;

        size_t x_start_h = oh * info.x_h / info.y_h;
        size_t x_end_h = ((oh + 1) * info.x_h + info.y_h - 1) / info.y_h;

        size_t x_start_w = ow * info.x_w / info.y_w;
        size_t x_end_w = ((ow + 1) * info.x_w + info.y_w - 1) / info.y_w;

        size_t count = (x_end_d - x_start_d) * (x_end_h - x_start_h) * (x_end_w - x_start_w);

        size_t y_offset = n * y_strides[0] + c * y_strides[1] + od * y_strides[2] + oh * y_strides[3] + ow * y_strides[4];

        // 🔹 Accumulate
        if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {

            float sum = 0.0f;

            for (size_t id = x_start_d; id < x_end_d; id++) {
                for (size_t ih = x_start_h; ih < x_end_h; ih++) {
                    for (size_t iw = x_start_w; iw < x_end_w; iw++) {

                        size_t x_offset = n * info.x_strides[0] + c * info.x_strides[1] + id * info.x_strides[2] + ih * info.x_strides[3] + iw * info.x_strides[4];

                        sum += utils::cast<float>(x[x_offset]);
                    }
                }
            }

            y[y_offset] = utils::cast<Tdata>(sum / (float)count);

        } else {

            Tdata sum = (Tdata)0;

            for (size_t id = x_start_d; id < x_end_d; id++) {
                for (size_t ih = x_start_h; ih < x_end_h; ih++) {
                    for (size_t iw = x_start_w; iw < x_end_w; iw++) {

                        size_t x_offset = n * info.x_strides[0] + c * info.x_strides[1] + id * info.x_strides[2] + ih * info.x_strides[3] + iw * info.x_strides[4];

                        sum += x[x_offset];
                    }
                }
            }

            y[y_offset] = sum / (Tdata)count;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        return calculateAdaptiveAvgPool3D<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), reinterpret_cast<const fp16_t *>(x));
    case INFINI_DTYPE_F32:
        return calculateAdaptiveAvgPool3D<float>(_info, reinterpret_cast<float *>(y), reinterpret_cast<const float *>(x));
    case INFINI_DTYPE_F64:
        return calculateAdaptiveAvgPool3D<double>(_info, reinterpret_cast<double *>(y), reinterpret_cast<const double *>(x));
    case INFINI_DTYPE_BF16:
        return calculateAdaptiveAvgPool3D<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), reinterpret_cast<const bf16_t *>(x));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::adaptive_avg_pool3d::cpu
