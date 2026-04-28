#ifndef __DEQUANTIZE_AWQ_INFO_H__
#define __DEQUANTIZE_AWQ_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::dequantize_awq {

class DequantizeAWQInfo {
    DequantizeAWQInfo() = default;

public:
    int _in_features, _out_features, _num_groups;

    int in_features() const { return _in_features; }
    int out_features() const { return _out_features; }
    int num_groups() const { return _num_groups; }

    static utils::Result<DequantizeAWQInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t qweight_desc,
        infiniopTensorDescriptor_t scales_desc,
        infiniopTensorDescriptor_t zeros_desc) {

        int _in_features = qweight_desc->dim(0);
        int _out_features = qweight_desc->dim(1);
        int _num_groups = scales_desc->dim(0);

        return utils::Result<DequantizeAWQInfo>(DequantizeAWQInfo{
            _in_features,
            _out_features,
            _num_groups});
    }
};

} // namespace op::dequantize_awq

#endif // __DEQUANTIZE_AWQ_INFO_H__
