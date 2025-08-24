#ifndef __LINEAR_INFO_H__
#define __LINEAR_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::linear {

class LinearInfo {
    LinearInfo() = default;

public:
    // Input and output shapes
    size_t batch_size;          // Total number of elements before the last dimension
    size_t in_features;         // Input feature size
    size_t out_features;        // Output feature size
    bool has_bias;              // Whether bias is provided
    
    // Strides for efficient memory access
    ptrdiff_t input_batch_stride;
    ptrdiff_t input_feature_stride;
    ptrdiff_t output_batch_stride;
    ptrdiff_t output_feature_stride;
    ptrdiff_t weight_out_stride;
    ptrdiff_t weight_in_stride;
    ptrdiff_t bias_stride;

    static utils::Result<LinearInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t bias_desc) {

        LinearInfo info;

        // Check that input and output have at least 2 dimensions
        if (input_desc->ndim() < 2 || output_desc->ndim() < 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check that weight has exactly 2 dimensions
        if (weight_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Get feature dimensions
        info.in_features = input_desc->dim(input_desc->ndim() - 1);
        info.out_features = output_desc->dim(output_desc->ndim() - 1);

        // Check weight shape: (out_features, in_features)
        if (weight_desc->dim(0) != info.out_features || weight_desc->dim(1) != info.in_features) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check bias shape if provided
        info.has_bias = (bias_desc != nullptr);
        if (info.has_bias) {
            if (bias_desc->ndim() != 1 || bias_desc->dim(0) != info.out_features) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            info.bias_stride = bias_desc->stride(0);
        } else {
            info.bias_stride = 0;
        }

        // Calculate batch size (all dimensions except the last one)
        info.batch_size = 1;
        for (size_t i = 0; i < input_desc->ndim() - 1; ++i) {
            if (input_desc->dim(i) != output_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            info.batch_size *= input_desc->dim(i);
        }

        // Get strides
        info.input_feature_stride = input_desc->stride(input_desc->ndim() - 1);
        info.output_feature_stride = output_desc->stride(output_desc->ndim() - 1);
        info.weight_out_stride = weight_desc->stride(0);
        info.weight_in_stride = weight_desc->stride(1);

        // Calculate batch strides (stride to move to next batch element)
        if (input_desc->ndim() >= 2) {
            info.input_batch_stride = input_desc->stride(input_desc->ndim() - 2);
        } else {
            info.input_batch_stride = info.in_features * info.input_feature_stride;
        }

        if (output_desc->ndim() >= 2) {
            info.output_batch_stride = output_desc->stride(output_desc->ndim() - 2);
        } else {
            info.output_batch_stride = info.out_features * info.output_feature_stride;
        }

        return utils::Result<LinearInfo>(info);
    }
};

} // namespace op::linear

#endif // __LINEAR_INFO_H__