#ifndef __LINEAR_BACKWARDS_INFO_H__
#define __LINEAR_BACKWARDS_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"
#include <algorithm>

namespace op::linear_backwards {

class LinearBackwardsInfo {
    LinearBackwardsInfo() = default;

public:
    // Input and output shapes
    size_t batch_size;          // Total number of elements before the last dimension
    size_t in_features;         // Input feature size
    size_t out_features;        // Output feature size
    bool has_bias;              // Whether bias gradient is computed
    
    // Strides for grad_input = grad_output @ weight
    ptrdiff_t grad_output_batch_stride;
    ptrdiff_t grad_output_feature_stride;
    ptrdiff_t grad_input_batch_stride;
    ptrdiff_t grad_input_feature_stride;
    
    // Strides for grad_weight = grad_output.T @ input
    ptrdiff_t input_batch_stride;
    ptrdiff_t input_feature_stride;
    ptrdiff_t grad_weight_out_stride;
    ptrdiff_t grad_weight_in_stride;
    
    // Strides for grad_bias = sum(grad_output, dim=0)
    ptrdiff_t grad_bias_stride;
    
    // Forward pass weight strides (for grad_input computation)
    ptrdiff_t weight_out_stride;
    ptrdiff_t weight_in_stride;

    static utils::Result<LinearBackwardsInfo> create(
        infiniopTensorDescriptor_t grad_input_desc,
        infiniopTensorDescriptor_t grad_weight_desc,
        infiniopTensorDescriptor_t grad_bias_desc,
        infiniopTensorDescriptor_t grad_output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc) {

        LinearBackwardsInfo info;

        // Check that input and grad_output have at least 2 dimensions
        if (input_desc->ndim() < 2 || grad_output_desc->ndim() < 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check that weight has exactly 2 dimensions
        if (weight_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Get feature dimensions
        info.in_features = input_desc->dim(input_desc->ndim() - 1);
        info.out_features = grad_output_desc->dim(grad_output_desc->ndim() - 1);

        // Check weight shape: (out_features, in_features)
        if (weight_desc->dim(0) != info.out_features || weight_desc->dim(1) != info.in_features) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check grad_input shape if provided
        if (grad_input_desc && (grad_input_desc->ndim() != input_desc->ndim() ||
                                grad_input_desc->dim(grad_input_desc->ndim() - 1) != info.in_features)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check grad_weight shape
        if (grad_weight_desc && (grad_weight_desc->ndim() != 2 ||
                                 grad_weight_desc->dim(0) != info.out_features ||
                                 grad_weight_desc->dim(1) != info.in_features)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check grad_bias shape if provided
        info.has_bias = (grad_bias_desc != nullptr);
        if (info.has_bias) {
            if (grad_bias_desc->ndim() != 1 || grad_bias_desc->dim(0) != info.out_features) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            info.grad_bias_stride = grad_bias_desc->stride(0);
        } else {
            info.grad_bias_stride = 0;
        }

        // Calculate batch size (all dimensions except the last one)
        info.batch_size = 1;
        for (size_t i = 0; i < input_desc->ndim() - 1; ++i) {
            if (input_desc->dim(i) != grad_output_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            info.batch_size *= input_desc->dim(i);
        }

        // Get strides for grad_output
        info.grad_output_feature_stride = grad_output_desc->stride(grad_output_desc->ndim() - 1);
        if (grad_output_desc->ndim() >= 2) {
            info.grad_output_batch_stride = grad_output_desc->stride(grad_output_desc->ndim() - 2);
        } else {
            info.grad_output_batch_stride = info.out_features * info.grad_output_feature_stride;
        }

        // Get strides for grad_input
        if (grad_input_desc) {
            info.grad_input_feature_stride = grad_input_desc->stride(grad_input_desc->ndim() - 1);
            if (grad_input_desc->ndim() >= 2) {
                info.grad_input_batch_stride = grad_input_desc->stride(grad_input_desc->ndim() - 2);
            } else {
                info.grad_input_batch_stride = info.in_features * info.grad_input_feature_stride;
            }
        } else {
            info.grad_input_feature_stride = 0;
            info.grad_input_batch_stride = 0;
        }

        // Get strides for input
        info.input_feature_stride = input_desc->stride(input_desc->ndim() - 1);
        if (input_desc->ndim() >= 2) {
            info.input_batch_stride = input_desc->stride(input_desc->ndim() - 2);
        } else {
            info.input_batch_stride = info.in_features * info.input_feature_stride;
        }

        // Get strides for weight and grad_weight
        info.weight_out_stride = weight_desc->stride(0);
        info.weight_in_stride = weight_desc->stride(1);
        
        if (grad_weight_desc) {
            info.grad_weight_out_stride = grad_weight_desc->stride(0);
            info.grad_weight_in_stride = grad_weight_desc->stride(1);
        } else {
            info.grad_weight_out_stride = 0;
            info.grad_weight_in_stride = 0;
        }

        return utils::Result<LinearBackwardsInfo>(info);
    }
};

} // namespace op::linear_backwards

#endif // __LINEAR_BACKWARDS_INFO_H__