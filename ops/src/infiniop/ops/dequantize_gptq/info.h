#ifndef __DEQUANTIZE_GPTQ_INFO_H__
#define __DEQUANTIZE_GPTQ_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

#include <cassert>

namespace op::dequantize_gptq {

class DequantizeGPTQInfo {
    DequantizeGPTQInfo() = default;

public:
    int _in_features, _out_features, _num_groups, _out_packed, _in_packed;

    int in_features() const { return _in_features; }
    int out_features() const { return _out_features; }
    int num_groups() const { return _num_groups; }
    int out_packed() const { return _out_packed; }
    int in_packed() const { return _in_packed; }

    static utils::Result<DequantizeGPTQInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t qweight_desc,
        infiniopTensorDescriptor_t scales_desc,
        infiniopTensorDescriptor_t zeros_desc,
        infiniopTensorDescriptor_t g_idx_desc) {

        const int _in_features = g_idx_desc->dim(0);    // real input channels
        const int _in_packed = qweight_desc->dim(0);    // ceil(in_features / 8)
        const int _out_features = qweight_desc->dim(1); // real output channels
        const int _num_groups = scales_desc->dim(0);    // should be in_features / group_size
        const int _out_packed = zeros_desc->dim(1);     // ceil(out_features / 8)

        assert(out_desc->dim(0) == _in_features);
        assert(out_desc->dim(1) == _out_features);
        assert(_in_packed == (_in_features + 7) / 8);
        assert(scales_desc->dim(1) == _out_features);
        assert(_num_groups == zeros_desc->dim(0));
        assert(_out_packed == (_out_features + 7) / 8);

        return utils::Result<DequantizeGPTQInfo>(
            DequantizeGPTQInfo{_in_features, _out_features, _num_groups, _out_packed, _in_packed});
    }
};

} // namespace op::dequantize_gptq

#endif // __DEQUANTIZE_GPTQ_INFO_H__
