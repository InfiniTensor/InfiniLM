#include "triplet_margin_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::triplet_margin_loss::cpu {

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
    infiniopTensorDescriptor_t anchor_desc,
    infiniopTensorDescriptor_t positive_desc,
    infiniopTensorDescriptor_t negative_desc,
    float margin,
    int p,
    float eps,
    int swap,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    // 创建 Info 对象
    auto result = TripletMarginLossInfo::create(out_desc, anchor_desc, positive_desc, negative_desc, margin, p, eps, swap, reduction);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        new Opaque(),
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 辅助函数：计算两个向量之间的 p-范数距离
template <typename T>
inline float compute_distance(const T *x, const T *y, size_t D, int p, float eps) {
    float sum = 0.0f;
    for (size_t i = 0; i < D; ++i) {
        float diff = std::abs(utils::cast<float>(x[i]) - utils::cast<float>(y[i]));
        if (p == 1) {
            sum += diff;
        } else if (p == 2) {
            sum += diff * diff;
        } else {
            sum += std::pow(diff, static_cast<float>(p));
        }
    }

    if (p == 1) {
        return sum + eps;
    } else if (p == 2) {
        // 标准 TripletMarginLoss 在 p=2 时通常加上 eps 再开方
        return std::sqrt(sum + eps);
    } else {
        return std::pow(sum + eps, 1.0f / static_cast<float>(p));
    }
}

template <typename T>
void calculate_cpu_impl(
    const TripletMarginLossInfo &info,
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative) {

    size_t N = info.batch_size();
    size_t D = info.feature_dim();
    float margin = info.margin();
    int p = info.p();
    float eps = info.eps();
    bool swap = info.swap();
    int reduction = info.reduction();

    auto out_ptr = reinterpret_cast<T *>(output);
    auto anc_ptr = reinterpret_cast<const T *>(anchor);
    auto pos_ptr = reinterpret_cast<const T *>(positive);
    auto neg_ptr = reinterpret_cast<const T *>(negative);

    // Reduction == 0: None
    if (reduction == 0) {
#pragma omp parallel for schedule(static)
        for (ptrdiff_t n = 0; n < (ptrdiff_t)N; ++n) {
            const T *a_row = anc_ptr + n * D;
            const T *p_row = pos_ptr + n * D;
            const T *n_row = neg_ptr + n * D;

            float dist_pos = compute_distance(a_row, p_row, D, p, eps);
            float dist_neg = compute_distance(a_row, n_row, D, p, eps);

            if (swap) {
                float dist_swap = compute_distance(p_row, n_row, D, p, eps);
                if (dist_swap < dist_neg) {
                    dist_neg = dist_swap;
                }
            }

            // loss = max(0, dist_pos - dist_neg + margin)
            float loss = std::max(0.0f, dist_pos - dist_neg + margin);
            out_ptr[n] = utils::cast<T>(loss);
        }
    }
    // Reduction != 0: Mean or Sum
    else {
        double total_loss = 0.0;

#pragma omp parallel for reduction(+ : total_loss) schedule(static)
        for (ptrdiff_t n = 0; n < (ptrdiff_t)N; ++n) {
            const T *a_row = anc_ptr + n * D;
            const T *p_row = pos_ptr + n * D;
            const T *n_row = neg_ptr + n * D;

            float dist_pos = compute_distance(a_row, p_row, D, p, eps);
            float dist_neg = compute_distance(a_row, n_row, D, p, eps);

            if (swap) {
                float dist_swap = compute_distance(p_row, n_row, D, p, eps);
                if (dist_swap < dist_neg) {
                    dist_neg = dist_swap;
                }
            }

            float loss = std::max(0.0f, dist_pos - dist_neg + margin);
            total_loss += static_cast<double>(loss);
        }

        if (reduction == 1) { // Mean
            total_loss /= static_cast<double>(N);
        }

        out_ptr[0] = utils::cast<T>(static_cast<float>(total_loss));
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative,
    void *stream) const {

    auto dtype = _info.dtype();

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, output, anchor, positive, negative);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::triplet_margin_loss::cpu
