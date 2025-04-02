#include "causal_softmax_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::causal_softmax::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc) {
    auto result = CausalSoftmaxInfo::create(y_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t causal_softmax(const CausalSoftmaxInfo *info, T *data) {
#pragma omp parallel for
    for (ptrdiff_t index = 0; index < ptrdiff_t(info->batch_size * info->seq_len); index++) {
        size_t ind = index;
        size_t offset = 0;
        size_t i = (ind % info->seq_len);
        offset += (ind % info->seq_len) * info->stride_i;
        ind /= info->seq_len;
        offset += (ind % info->batch_size) * info->stride_b;
        for (size_t j = info->total_seq_len - info->seq_len + i + 1; j < info->total_seq_len; j++) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                data[offset + j * info->stride_j] = utils::cast<fp16_t>(0.0f);
            } else {
                data[offset + j * info->stride_j] = 0.0f;
            }
        }
        float val = op::common_cpu::reduce_op::max(&data[offset], info->total_seq_len - info->seq_len + i + 1, info->stride_j);
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                data[offset + j * info->stride_j] = utils::cast<fp16_t>(std::exp(utils::cast<float>(data[offset + j * info->stride_j]) - val));
            } else {
                data[offset + j * info->stride_j] = std::exp(data[offset + j * info->stride_j] - val);
            }
        }
        float sum = op::common_cpu::reduce_op::sum(&data[offset], info->total_seq_len - info->seq_len + i + 1, info->stride_j);
        for (size_t j = 0; j <= info->total_seq_len - info->seq_len + i; j++) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                data[offset + j * info->stride_j] = utils::cast<fp16_t>(utils::cast<float>(data[offset + j * info->stride_j]) / sum);
            } else {
                data[offset + j * info->stride_j] = data[offset + j * info->stride_j] / sum;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *data,
    void *stream) const {

    if (_info.dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(causal_softmax<fp16_t>(&_info, (fp16_t *)data));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(causal_softmax<float>(&_info, (float *)data));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::causal_softmax::cpu
