#include "huber_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <omp.h>

#include "../../../../utils/custom_types.h"

namespace op::huber_loss::cpu {

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
    float delta,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = HuberLossInfo::create(out_desc, input_desc, target_desc, delta, reduction);
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
    const HuberLossInfo &info,
    void *output,
    const void *input,
    const void *target) {

    size_t count = info.count();
    float delta = info.delta();
    int reduction = info.reduction();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    // Huber Loss 中 target 是数值，类型与 input 一致
    auto tar_ptr = reinterpret_cast<const T *>(target);
    float half_delta = 0.5f * delta;

    if (reduction == 0) { // None
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)count; ++i) {
            float val = utils::cast<float>(in_ptr[i]);
            float tgt = utils::cast<float>(tar_ptr[i]);

            float diff = val - tgt;
            float abs_diff = std::abs(diff);
            float loss = 0.0f;

            if (abs_diff < delta) {
                // 0.5 * (x - y)^2
                loss = 0.5f * diff * diff;
            } else {
                // delta * (|x - y| - 0.5 * delta)
                loss = delta * (abs_diff - half_delta);
            }

            out_ptr[i] = utils::cast<T>(loss);
        }
    } else { // Mean or Sum
        double total_loss = 0.0;

#pragma omp parallel for reduction(+ : total_loss) schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)count; ++i) {
            float val = utils::cast<float>(in_ptr[i]);
            float tgt = utils::cast<float>(tar_ptr[i]);

            float diff = val - tgt;
            float abs_diff = std::abs(diff);
            float loss = 0.0f;

            if (abs_diff < delta) {
                loss = 0.5f * diff * diff;
            } else {
                loss = delta * (abs_diff - half_delta);
            }

            total_loss += static_cast<double>(loss);
        }

        if (reduction == 1) { // Mean
            total_loss /= static_cast<double>(count);
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
    void *stream) const {

    auto dtype = _info.dtype();
    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, input, target);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, input, target);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, input, target);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, input, target);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::huber_loss::cpu
