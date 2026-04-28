#include "topkrouter_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include <algorithm>

namespace op::topkrouter::cpu {
Descriptor::~Descriptor() {
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t x_desc,
                                  infiniopTensorDescriptor_t correction_bias_desc) {
    auto result = TopkrouterInfo::create(x_desc);
    CHECK_RESULT(result);
    auto info = result.take();

    if (info.x_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(nullptr, std::move(info), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
inline float sigmoid_func(T x) {
    float value;
    if constexpr (std::is_same<T, fp16_t>::value) {
        value = _f16_to_f32(x);
    } else if constexpr (std::is_same<T, bf16_t>::value) {
        value = _bf16_to_f32(x);
    } else {
        value = x;
    }
    return 1.0f / (1.0f + std::exp(-value));
}

template <typename T>
void topkrouter_cpu_one_token(float *values_input,                                    // 输出数据
                              int *indices_input,                                     // 输出索引
                              const T *x_input,                                       // 输入数据
                              std::vector<std::pair<float, size_t>> &value_index_arr, // 输入数据
                              const float *correction_bias,
                              const float routed_scaling_factor, const size_t topk,
                              const size_t width, const size_t n_routed_experts, const size_t n_group,
                              const size_t topk_group, const bool norm_topk_prob) {

    // ------------------------------------------------------ //
    //               对输入数据做 sigmoid                       //
    // ------------------------------------------------------ //
    for (size_t i = 0; i < width; ++i) {
        value_index_arr[i].first = sigmoid_func(value_index_arr[i].first);
    }

    // ------------------------------------------------------ //
    //                   再加偏置                              //
    // ------------------------------------------------------ //
    for (size_t i = 0; i < width; ++i) {
        value_index_arr[i].first += correction_bias[i];
    }

    // ----------------------------------------------------------- //
    //          分为 n_group 组，找出每组的最大值                      //
    // ----------------------------------------------------------- //
    std::vector<std::pair<float, size_t>> value_index_group;
    value_index_group.resize(n_group);

    const size_t group_size = width / n_group; // group_size表示每个组的元素数量
    for (size_t igroup = 0; igroup < n_group; ++igroup) {
        std::vector<std::pair<float, size_t>> value_index_warp;
        value_index_warp.resize(group_size);

        auto it = value_index_arr.begin() + igroup * group_size;
        for (size_t i = 0; i < group_size; ++i) {
            value_index_warp[i] = {(it++)->first, i};
        }

        // 每个group中的数据，进行排序
        std::sort(value_index_warp.begin(), value_index_warp.end(),
                  [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) { return a.first > b.first; });

        // 取前两个的和，作为最大值
        value_index_group[igroup] = {value_index_warp[0].first + value_index_warp[1].first, igroup};
    }

    // ------------------------------------------------------------------ //
    //       对 value_index_group 的数据, 再选前 topk_group 个               //
    // ------------------------------------------------------------------ //
    std::sort(value_index_group.begin(), value_index_group.end(),
              [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) { return a.first > b.first; });

    std::vector<bool> group_mask;
    group_mask.resize(n_group, false);
    for (size_t i = 0; i < topk_group; ++i) {
        size_t index = value_index_group[i].second;
        group_mask[index] = true;
    }

    // ------------------------------------------------------------------ //
    //              根据group_mask，false的组的数值置0                       //
    // ------------------------------------------------------------------ //
    for (size_t igroup = 0; igroup < n_group; ++igroup) {
        if (group_mask[igroup]) {
            continue;
        }

        auto it = value_index_arr.begin() + igroup * group_size;
        for (size_t i = 0; i < group_size; ++i) {
            (it++)->first = 0.0f;
        }
    }

    // ------------------------------------------------------------------ //
    //                     最后整体做topk                                   //
    // ------------------------------------------------------------------ //
    std::sort(value_index_arr.begin(), value_index_arr.end(),
              [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) { return a.first > b.first; });

    // ----------------------------------------------------------- //
    //                取topk个数据                                  //
    // ----------------------------------------------------------- //
    float exp_sum = 1e-9f;
    for (size_t i = 0; i < topk; ++i) {
        size_t index = value_index_arr[i].second;
        float exp_value = sigmoid_func(x_input[index]);

        values_input[i] = exp_value;
        indices_input[i] = static_cast<int>(index);

        exp_sum += exp_value;
    }

    // ----------------------------------------------------------- //
    //                    归一化                                    //
    // ----------------------------------------------------------- //
    if (norm_topk_prob) {
        for (size_t i = 0; i < topk; ++i) {
            values_input[i] = routed_scaling_factor * values_input[i] / exp_sum;
        }
    }
}

template <typename T>
infiniStatus_t topkrouter_cpu_func(float *values, int *indices,
                                   const T *x,
                                   const float *correction_bias, const float routed_scaling_factor, const size_t topk,
                                   const size_t N, const size_t width, const size_t n_routed_experts = 256,
                                   const size_t n_group = 8, const size_t topk_group = 4,
                                   const bool norm_topk_prob = true) {
    /*
    O-----------> width 地址连续
    |
    |
    N
    */
    for (size_t n = 0; n < N; ++n) {
        float *values_input = values + n * topk;
        int *indices_input = indices + n * topk;
        const T *x_input = x + n * width;

        std::vector<std::pair<float, size_t>> value_index_arr;
        value_index_arr.resize(width);

        for (size_t i = 0; i < width; ++i) {
            float temp;
            if constexpr (std::is_same<T, fp16_t>::value) {
                temp = _f16_to_f32(x_input[i]);
            } else if constexpr (std::is_same<T, bf16_t>::value) {
                temp = _bf16_to_f32(x_input[i]);
            } else {
                temp = x_input[i];
            }
            value_index_arr[i] = {temp, i};
        }

        topkrouter_cpu_one_token<T>(values_input, indices_input, x_input, value_index_arr,
                                    correction_bias, routed_scaling_factor,
                                    topk, width, n_routed_experts, n_group, topk_group, norm_topk_prob);
    }

    return INFINI_STATUS_SUCCESS;
} // namespace op::topkrouter::cpu

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, float *values, int *indices, const void *x,
                                     const float *correction_bias, const float routed_scaling_factor, const size_t topk,
                                     void *stream) const {
    size_t N = _info.N;
    size_t width = _info.width;

    // 下面是 deepseek的config.json的超参数
    const size_t n_routed_experts = 256;
    const size_t n_group = 8;
    const size_t topk_group = 4;
    const bool norm_topk_prob = true;

    if ((width != n_routed_experts) || (width % n_group != 0) || (256 != width)) {
        return INFINI_STATUS_BAD_PARAM;
    }

    if (_info.xtype == INFINI_DTYPE_F32) {
        topkrouter_cpu_func(values, indices, static_cast<const float *>(x), correction_bias, routed_scaling_factor, topk, N,
                            width, n_routed_experts, n_group, topk_group, norm_topk_prob);
    } else if (_info.xtype == INFINI_DTYPE_F16) {
        topkrouter_cpu_func(values, indices, static_cast<const fp16_t *>(x), correction_bias, routed_scaling_factor, topk, N,
                            width, n_routed_experts, n_group, topk_group, norm_topk_prob);
    } else if (_info.xtype == INFINI_DTYPE_BF16) {
        topkrouter_cpu_func(values, indices, static_cast<const bf16_t *>(x), correction_bias, routed_scaling_factor, topk, N,
                            width, n_routed_experts, n_group, topk_group, norm_topk_prob);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::topkrouter::cpu
