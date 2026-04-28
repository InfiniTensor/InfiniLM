#include "triplet_margin_with_distance_loss_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <omp.h>
#include <vector>

#include "../../../../utils/custom_types.h"

namespace op::triplet_margin_with_distance_loss::cpu {

struct Descriptor::Opaque {
    size_t batch_size;
    size_t feature_dim;
};

Descriptor::~Descriptor() {
    if (_opaque) {
        delete _opaque;
        _opaque = nullptr;
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t anchor_desc,
    infiniopTensorDescriptor_t positive_desc,
    infiniopTensorDescriptor_t negative_desc,
    float margin,
    int swap,
    int reduction) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);

    auto result = TripletMarginWithDistanceLossInfo::create(
        output_desc, anchor_desc, positive_desc, negative_desc, margin, swap, reduction);
    CHECK_RESULT(result);

    // 解析形状信息
    size_t ndim = anchor_desc->ndim();
    size_t feature_dim = (ndim > 0) ? anchor_desc->shape()[ndim - 1] : 1;
    size_t total_elements = result->num_elements();
    size_t batch_size = total_elements / feature_dim;

    auto opaque = new Opaque();
    opaque->batch_size = batch_size;
    opaque->feature_dim = feature_dim;

    *desc_ptr = new Descriptor(
        opaque,
        result.take(),
        0,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// 辅助函数：计算两个向量的欧氏距离
template <typename T>
inline float compute_pairwise_distance(const T *x, const T *y, size_t len, float eps = 1e-6f) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < len; ++i) {
        float diff = utils::cast<float>(x[i]) - utils::cast<float>(y[i]);
        sum_sq += diff * diff;
    }
    return std::sqrt(sum_sq + eps);
}

// FIX: 移除了 Descriptor::Opaque* 参数，改为直接传入 batch_size 和 feature_dim
template <typename T>
void calculate_cpu_impl(
    const TripletMarginWithDistanceLossInfo &info,
    size_t batch_size,
    size_t feature_dim,
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative) {

    auto out_ptr = reinterpret_cast<T *>(output);
    auto a_ptr = reinterpret_cast<const T *>(anchor);
    auto p_ptr = reinterpret_cast<const T *>(positive);
    auto n_ptr = reinterpret_cast<const T *>(negative);

    float margin = info.margin();
    bool swap = info.swap();
    int reduction = info.reduction(); // 0:None, 1:Mean, 2:Sum

    float total_loss = 0.0f;

#pragma omp parallel for schedule(static) reduction(+ : total_loss)
    for (ptrdiff_t i = 0; i < (ptrdiff_t)batch_size; ++i) {
        size_t offset = i * feature_dim;

        const T *curr_a = a_ptr + offset;
        const T *curr_p = p_ptr + offset;
        const T *curr_n = n_ptr + offset;

        float dist_pos = compute_pairwise_distance(curr_a, curr_p, feature_dim);
        float dist_neg = compute_pairwise_distance(curr_a, curr_n, feature_dim);

        if (swap) {
            float dist_pn = compute_pairwise_distance(curr_p, curr_n, feature_dim);
            if (dist_pn < dist_neg) {
                dist_neg = dist_pn;
            }
        }

        float loss = std::max(dist_pos - dist_neg + margin, 0.0f);

        if (reduction == 0) {
            out_ptr[i] = utils::cast<T>(loss);
        } else {
            total_loss += loss;
        }
    }

    if (reduction != 0) {
        if (reduction == 1) { // Mean
            total_loss /= static_cast<float>(batch_size);
        }
        out_ptr[0] = utils::cast<T>(total_loss);
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
    // 从 _opaque 中获取形状参数
    size_t batch_size = _opaque->batch_size;
    size_t feature_dim = _opaque->feature_dim;

    switch (dtype) {
    case INFINI_DTYPE_F32:
        cpu::calculate_cpu_impl<float>(_info, batch_size, feature_dim, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_F64:
        cpu::calculate_cpu_impl<double>(_info, batch_size, feature_dim, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_F16:
        cpu::calculate_cpu_impl<fp16_t>(_info, batch_size, feature_dim, output, anchor, positive, negative);
        break;
    case INFINI_DTYPE_BF16:
        cpu::calculate_cpu_impl<bf16_t>(_info, batch_size, feature_dim, output, anchor, positive, negative);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::triplet_margin_with_distance_loss::cpu
