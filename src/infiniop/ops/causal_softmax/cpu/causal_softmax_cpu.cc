#include "causal_softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::causal_softmax::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto result = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t causal_softmax(const CausalSoftmaxInfo *info, T *y, const T *x) {
#pragma omp parallel for
    for (ptrdiff_t index = 0; index < ptrdiff_t(info->batch_size * info->seq_len); index++) {
        size_t batch = index / info->seq_len;
        size_t i = (index % info->seq_len);
        ptrdiff_t y_offset = batch * info->y_stride_b + i * info->y_stride_i;
        ptrdiff_t x_offset = batch * info->x_stride_b + i * info->x_stride_i;
        T *y_ = y + y_offset;
        const T *x_ = x + x_offset;

        for (size_t j = info->total_seq_len - info->seq_len + i + 1; j < info->total_seq_len; j++) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                y_[j * info->y_stride_j] = utils::cast<fp16_t>(0.0f);
            } else {
                y_[j * info->y_stride_j] = 0.0f;
            }
        }
        float val = op::common_cpu::reduce_op::max(x_, info->total_seq_len - info->seq_len + i + 1, info->x_stride_j);
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                y_[j * info->y_stride_j] = utils::cast<fp16_t>(std::exp(utils::cast<float>(x_[j * info->x_stride_j]) - val));
            } else {
                y_[j * info->y_stride_j] = std::exp(x_[j * info->x_stride_j] - val);
            }
        }
        float sum = op::common_cpu::reduce_op::sum(y_, info->total_seq_len - info->seq_len + i + 1, info->y_stride_j);
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                y_[j * info->y_stride_j] = utils::cast<fp16_t>(utils::cast<float>(y_[j * info->y_stride_j]) / sum);
            } else {
                y_[j * info->y_stride_j] = y_[j * info->y_stride_j] / sum;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) const {

    if (_info.dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(causal_softmax<fp16_t>(&_info, (fp16_t *)y, (const fp16_t *)x));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(causal_softmax<float>(&_info, (float *)y, (const float *)x));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::causal_softmax::cpu
