#include "smooth_l1_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>

#include "../../../../utils/custom_types.h"

namespace op::smooth_l1_loss::cpu {

Descriptor::~Descriptor() = default;

// ==================================================================
// 创建描述符
// ==================================================================
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    float beta,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = SmoothL1LossInfo::create(out_desc, input_desc, target_desc, beta, reduction);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        nullptr,
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// ==================================================================
// 核心计算逻辑
// ==================================================================
template <typename T>
void calculate_cpu_impl(
    const SmoothL1LossInfo &info,
    void *output,
    const void *input,
    const void *target) {

    size_t numel = info.numel();
    float beta = info.beta();
    int reduction = info.reduction();

    float inv_beta = (beta > 0) ? (1.0f / beta) : 0.0f;
    float half_beta = 0.5f * beta;

    auto out_ptr = reinterpret_cast<T *>(output);
    auto in_ptr = reinterpret_cast<const T *>(input);
    auto tar_ptr = reinterpret_cast<const T *>(target);

    // ----------------------------------------------------
    // 模式 A: Elementwise (None)
    // ----------------------------------------------------
    if (reduction == 0) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)numel; ++i) {
            float in_val = utils::cast<float>(in_ptr[i]);
            float tar_val = utils::cast<float>(tar_ptr[i]);

            float diff = std::abs(in_val - tar_val);
            float loss;

            if (diff < beta) {
                loss = 0.5f * diff * diff * inv_beta;
            } else {
                loss = diff - half_beta;
            }

            // [核心] 计算完 float 后，转回目标类型 T 存储
            out_ptr[i] = utils::cast<T>(loss);
        }
    }
    // ----------------------------------------------------
    // 模式 B: Reduction (Mean / Sum)
    // ----------------------------------------------------
    else {
        double total_sum = 0.0;

#pragma omp parallel for reduction(+ : total_sum) schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)numel; ++i) {
            float in_val = utils::cast<float>(in_ptr[i]);
            float tar_val = utils::cast<float>(tar_ptr[i]);

            float diff = std::abs(in_val - tar_val);
            float loss;

            if (diff < beta) {
                loss = 0.5f * diff * diff * inv_beta;
            } else {
                loss = diff - half_beta;
            }

            total_sum += static_cast<double>(loss);
        }

        if (reduction == 1) { // Mean
            total_sum /= static_cast<double>(numel);
        }

        // Reduction 结果写入第 0 个位置
        out_ptr[0] = utils::cast<T>(static_cast<float>(total_sum));
    }
}

// ==================================================================
// 分发逻辑
// ==================================================================
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *target,
    void *stream) const {

    auto dtype = _info.dtype();

#define DISPATCH_TYPE(T)                                      \
    cpu::calculate_cpu_impl<T>(_info, output, input, target); \
    return INFINI_STATUS_SUCCESS;

    switch (dtype) {
    case INFINI_DTYPE_F32:
        DISPATCH_TYPE(float);
    case INFINI_DTYPE_F64:
        DISPATCH_TYPE(double);
    case INFINI_DTYPE_F16:
        DISPATCH_TYPE(fp16_t);
    case INFINI_DTYPE_BF16:
        DISPATCH_TYPE(bf16_t);

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef DISPATCH_TYPE
}

} // namespace op::smooth_l1_loss::cpu
