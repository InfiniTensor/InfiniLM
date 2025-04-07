#include "rms_norm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::rms_norm::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t rmsnorm(const RMSNormInfo *info, T *y, const T *x, const T *w) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ptrdiff_t(info->shape[0]); i++) {
        T *x_ = (T *)(x + i * info->x_strides[0]);
        T *y_ = (T *)(y + i * info->y_strides[0]);

        // [Reduce] sum of x^2 on last dimension
        T ss = op::common_cpu::reduce_op::sumSquared(x_, info->shape[1], info->x_strides[1]);

        // 1 / (sqrt(sum/dim + eps))
        T rms = (T)1 / std::sqrt(ss / (T)(info->shape[1]) + (T)(info->epsilon));

        for (size_t j = 0; j < info->shape[1]; j++) {
            y_[j * info->y_strides[1]] = x_[j * info->x_strides[1]] * w[j] * rms;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

template <typename Tw>
infiniStatus_t rmsnormF16(const RMSNormInfo *info, fp16_t *y, const fp16_t *x, const Tw *w) {
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < ptrdiff_t(info->shape[0]); i++) {
        fp16_t *x_ = (fp16_t *)(x + i * info->x_strides[0]);
        fp16_t *y_ = (fp16_t *)(y + i * info->y_strides[0]);

        // [Reduce] sum of x^2 on last dimension
        float ss = op::common_cpu::reduce_op::sumSquared(x_, info->shape[1], info->x_strides[1]);

        // 1 / (sqrt(sum/dim + eps))
        float rms = 1.f / std::sqrt(ss / (float)(info->shape[1]) + info->epsilon);

        for (size_t j = 0; j < info->shape[1]; j++) {
            if constexpr (std::is_same<Tw, float>::value) {
                float val = utils::cast<float>(x_[j * info->x_strides[1]]) * w[j] * rms;
                y_[j * info->y_strides[1]] = utils::cast<fp16_t>(val);
            } else if constexpr (std::is_same<Tw, fp16_t>::value) {
                float val = utils::cast<float>(x_[j * info->x_strides[1]]) * utils::cast<float>(w[j]) * rms;
                y_[j * info->y_strides[1]] = utils::cast<fp16_t>(val);
            } else {
                std::abort();
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y, const void *x, const void *w,
    void *stream) const {
    if (_info.atype == INFINI_DTYPE_F16) {
        if (_info.wtype == INFINI_DTYPE_F16) {
            CHECK_STATUS(rmsnormF16(&_info, (fp16_t *)y, (const fp16_t *)x, (const fp16_t *)w));
        } else if (_info.wtype == INFINI_DTYPE_F32) {
            CHECK_STATUS(rmsnormF16(&_info, (fp16_t *)y, (const fp16_t *)x, (const float *)w));
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (_info.atype == INFINI_DTYPE_F32) {
        CHECK_STATUS(rmsnorm(&_info, (float *)y, (float *)x, (float *)w));
    } else if (_info.atype == INFINI_DTYPE_F64) {
        CHECK_STATUS(rmsnorm(&_info, (double *)y, (double *)x, (double *)w));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::rms_norm::cpu
