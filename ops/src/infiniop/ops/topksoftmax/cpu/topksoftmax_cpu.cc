#include "topksoftmax_cpu.h"
#include "../../../../utils.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"
#include "topksoftmax_cpu.h"
#include <algorithm>

namespace op::topksoftmax::cpu {
Descriptor::~Descriptor() {
}

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t x_desc) {
    auto result = TopksoftmaxInfo::create(x_desc);
    CHECK_RESULT(result);

    auto info = result.take();
    if (info.x_strides[1] != 1) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    *desc_ptr = new Descriptor(nullptr, std::move(info), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

void topksoftmax_cpu_one_token(float *values_input,                                    // 输出数据
                               int *indices_input,                                     // 输出索引
                               std::vector<std::pair<float, size_t>> &value_index_arr, // 输入数据
                               size_t topk,
                               bool norm,
                               size_t width) {

    // ------------------------------------------------ //
    //             第一步：计算最大值                       //
    // ------------------------------------------------ //
    float value_max = value_index_arr[0].first;
    for (size_t i = 1; i < width; ++i) {
        value_max = value_index_arr[i].first > value_max ? value_index_arr[i].first : value_max;
    }

    // ------------------------------------------------ //
    //             第二步： 指数计算                       //
    // ------------------------------------------------ //
    float exp_sum = 0.0f;
    for (size_t i = 0; i < width; ++i) {
        float value = std::exp(value_index_arr[i].first - value_max);
        value_index_arr[i].first = value;
        exp_sum += value;
    }

    // ------------------------------------------------ //
    //              第三步：计算 Softmax                  //
    // ------------------------------------------------ //
    for (size_t i = 0; i < width; ++i) {
        value_index_arr[i].first /= exp_sum;
    }

    // ------------------------------------------------ //
    //           第四步：计算 排序                        //
    // ------------------------------------------------ //
    std::sort(value_index_arr.begin(), value_index_arr.end(),
              [](const std::pair<float, size_t> &a, const std::pair<float, size_t> &b) { return a.first > b.first; });

    // ------------------------------------------------ //
    //           第五步：   topk                         //
    // ------------------------------------------------ //
    exp_sum = 0.0f;
    for (size_t i = 0; i < topk; ++i) {
        values_input[i] = value_index_arr[i].first;
        indices_input[i] = static_cast<int>(value_index_arr[i].second);
        exp_sum += values_input[i];
    }

    // ------------------------------------------------ //
    //           第6步： norm归一化                       //
    // ------------------------------------------------ //
    if (norm) {
        for (size_t i = 0; i < topk; ++i) {
            values_input[i] /= exp_sum;
        }
    }
}

template <typename T>
infiniStatus_t topksoftmax_cpu_func(float *values, int *indices,
                                    const T *x,
                                    size_t topk, bool norm, size_t N, size_t width) {
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

        // ------------------------------------------------ //
        //             第0步： 数据先转换到 float              //
        // ------------------------------------------------ //
        float temp;
        for (size_t i = 0; i < width; ++i) {
            if constexpr (std::is_same<T, fp16_t>::value) {
                temp = _f16_to_f32(x_input[i]);
            } else if constexpr (std::is_same<T, bf16_t>::value) {
                temp = _bf16_to_f32(x_input[i]);
            } else {
                temp = x_input[i];
            }
            value_index_arr[i] = {temp, i};
        }

        topksoftmax_cpu_one_token(values_input,
                                  indices_input,
                                  value_index_arr,
                                  topk,
                                  norm,
                                  width);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, float *values, int *indices, const void *x,
                                     const size_t topk, const bool norm, void *stream) const {
    size_t N = _info.N;
    size_t width = _info.width;
    if (_info.xtype == INFINI_DTYPE_F32) {
        topksoftmax_cpu_func<float>(values, indices, (const float *)x, topk, norm, N, width);
    } else if (_info.xtype == INFINI_DTYPE_F16) {
        topksoftmax_cpu_func<fp16_t>(values, indices, (const fp16_t *)x, topk, norm, N, width);
    } else if (_info.xtype == INFINI_DTYPE_BF16) {
        topksoftmax_cpu_func<bf16_t>(values, indices, (const bf16_t *)x, topk, norm, N, width);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::topksoftmax::cpu
