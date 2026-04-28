#ifndef __ADAPTIVE_AVG_POOL1D_INFO_H__
#define __ADAPTIVE_AVG_POOL1D_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::adaptive_avg_pool1d {

class AdaptiveAvgPool1dInfo {
    AdaptiveAvgPool1dInfo() = default;

public:
    size_t _input_size;
    size_t _output_size;
    size_t _num_channels;
    int _dtype;

    size_t input_size() const { return _input_size; }
    size_t output_size() const { return _output_size; }
    size_t num_channels() const { return _num_channels; }
    int dtype() const { return _dtype; }

    static utils::Result<AdaptiveAvgPool1dInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc) {

        // 1. 检查数据类型一致性
        if (out_desc->dtype() != in_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        // 2. 检查维度 (至少 2 维: C, L 或 N, C, L)
        size_t ndim = in_desc->ndim();
        if (ndim < 2 || out_desc->ndim() != ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t num_channels = 1;
        for (size_t i = 0; i < ndim - 1; ++i) {
            if (in_desc->shape()[i] != out_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            num_channels *= in_desc->shape()[i];
        }

        // 4. 获取输入和输出的长度 (L)
        size_t input_size = in_desc->shape()[ndim - 1];
        size_t output_size = out_desc->shape()[ndim - 1];
        int dtype = in_desc->dtype();

        return utils::Result<AdaptiveAvgPool1dInfo>(AdaptiveAvgPool1dInfo{
            input_size,
            output_size,
            num_channels,
            dtype});
    }
};

} // namespace op::adaptive_avg_pool1d

#endif // __ADAPTIVE_AVG_POOL1D_INFO_H__
