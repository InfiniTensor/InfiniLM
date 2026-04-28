#include "inner_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::inner::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t other_desc) {

    auto result = InnerInfo::create(out_desc, input_desc, other_desc);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace {

template <typename T>
infiniStatus_t inner(const InnerInfo *info, const T *input, const T *other, T *out) {
#pragma omp parallel for
    for (ptrdiff_t out_index = 0; out_index < ptrdiff_t(info->total_elements); out_index++) {

        size_t out_dim_pos = info->out_ndim - 1;
        size_t out_offset = op::common_cpu::indexToOffset(out_index, info->out_ndim, info->out_shape.data(), info->out_strides.data());
        size_t index = out_index;

        ptrdiff_t input_offset = 0;
        ptrdiff_t other_offset = 0;

        for (int i = (int)info->other_ndim - 2; i >= 0; i--) {
            other_offset += (index % info->out_shape[out_dim_pos]) * info->other_strides[i];
            index /= info->out_shape[out_dim_pos--];
        }
        for (int i = (int)info->input_ndim - 2; i >= 0; i--) {
            input_offset += (index % info->out_shape[out_dim_pos]) * info->input_strides[i];
            index /= info->out_shape[out_dim_pos--];
        }

        float tmp = 0.;
        for (size_t i = 0; i < info->oper_len; i++) {
            if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
                tmp += utils::cast<float>(input[input_offset]) * utils::cast<float>(other[other_offset]);
            } else {
                tmp += input[input_offset] * other[other_offset];
            }
            input_offset += info->input_strides[info->input_ndim - 1];
            other_offset += info->other_strides[info->other_ndim - 1];
        }

        if constexpr (std::is_same<T, fp16_t>::value || std::is_same<T, bf16_t>::value) {
            out[out_offset] = utils::cast<T>(tmp);
        } else {
            out[out_offset] = tmp;
        }
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out,
    const void *input,
    const void *other,
    void *stream) const {

    if (_info.dtype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(inner(&_info, (const bf16_t *)input, (const bf16_t *)other, (bf16_t *)out));
    } else if (_info.dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(inner(&_info, (const fp16_t *)input, (const fp16_t *)other, (fp16_t *)out));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(inner(&_info, (const float *)input, (const float *)other, (float *)out));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::inner::cpu
