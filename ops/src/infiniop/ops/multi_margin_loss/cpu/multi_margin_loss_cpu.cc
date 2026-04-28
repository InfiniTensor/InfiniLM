#include "multi_margin_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

#include "../../../../utils/custom_types.h"

namespace op::multi_margin_loss::cpu {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t weight_desc,
    int p,
    float margin,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = MultiMarginLossInfo::create(out_desc, input_desc, target_desc, weight_desc, p, margin, reduction);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void calculate_cpu_impl(
    const MultiMarginLossInfo &info,
    void *output,
    const void *input,
    const void *target,
    const void *weight) {

    size_t N = info.batch_size();
    size_t C = info.num_classes();
    int p = info.p();
    float margin = info.margin();
    int reduction = info.reduction();
    bool has_weight = info.has_weight();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto tar_ptr = reinterpret_cast<const int64_t *>(target);
    auto weight_ptr = reinterpret_cast<const T *>(weight);

    if (reduction == 0) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t n = 0; n < (ptrdiff_t)N; ++n) {
            int64_t target_idx = tar_ptr[n];

            if (target_idx < 0 || target_idx >= static_cast<int64_t>(C)) {
                out_ptr[n] = utils::cast<T>(0.0f);
                continue;
            }

            const T *row_ptr = in_ptr + n * C;
            float target_score = utils::cast<float>(row_ptr[target_idx]);
            float sum_loss = 0.0f;

            for (size_t c = 0; c < C; ++c) {
                if (c == static_cast<size_t>(target_idx)) {
                    continue;
                }

                float other_score = utils::cast<float>(row_ptr[c]);
                float diff = margin - target_score + other_score;

                if (diff > 0.0f) {
                    sum_loss += (p == 1) ? diff : (diff * diff);
                }
            }

            sum_loss /= static_cast<float>(C);

            if (has_weight) {
                float w = utils::cast<float>(weight_ptr[target_idx]);
                sum_loss *= w;
            }

            out_ptr[n] = utils::cast<T>(sum_loss);
        }
    } else {
        double total_loss = 0.0;

#pragma omp parallel for reduction(+ : total_loss) schedule(static)
        for (ptrdiff_t n = 0; n < (ptrdiff_t)N; ++n) {
            int64_t target_idx = tar_ptr[n];

            if (target_idx < 0 || target_idx >= static_cast<int64_t>(C)) {
                continue;
            }

            const T *row_ptr = in_ptr + n * C;
            float target_score = utils::cast<float>(row_ptr[target_idx]);
            float sum_sample_loss = 0.0f;

            for (size_t c = 0; c < C; ++c) {
                if (c == static_cast<size_t>(target_idx)) {
                    continue;
                }

                float other_score = utils::cast<float>(row_ptr[c]);
                float diff = margin - target_score + other_score;

                if (diff > 0.0f) {
                    sum_sample_loss += (p == 1) ? diff : (diff * diff);
                }
            }

            sum_sample_loss /= static_cast<float>(C);

            if (has_weight) {
                float w = utils::cast<float>(weight_ptr[target_idx]);
                sum_sample_loss *= w;
            }

            total_loss += static_cast<double>(sum_sample_loss);
        }

        if (reduction == 1) {
            total_loss /= static_cast<double>(N);
        }

        out_ptr[0] = utils::cast<T>(static_cast<float>(total_loss));
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *target,
    const void *weight,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input, target, weight);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input, target, weight);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input, target, weight);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input, target, weight);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::multi_margin_loss::cpu
