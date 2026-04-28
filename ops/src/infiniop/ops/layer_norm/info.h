#ifndef __LAYER_NORM_INFO_H__
#define __LAYER_NORM_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::layer_norm {

class LayerNormInfo {
private:
    LayerNormInfo() = default;

public:
    infiniDtype_t dtype;
    size_t ndim;
    std::vector<size_t> input_shape;
    size_t normalized_size;
    size_t othersize;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> input_standardization_strides;
    std::vector<ptrdiff_t> input_std_deviation_strides;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> weight_strides;
    std::vector<ptrdiff_t> bias_strides;
    float eps;
    bool bias_exist;

    static utils::Result<LayerNormInfo> createLayerNormInfo(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_standardization_desc,
        infiniopTensorDescriptor_t input_std_deviation_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t weight_desc,
        infiniopTensorDescriptor_t bias_desc,
        float eps) {

        CHECK_SAME_SHAPE(
            output_desc->shape(), input_desc->shape(), input_standardization_desc->shape());
        size_t ndim = input_desc->ndim();
        size_t normalized_size = input_desc->dim(ndim - 1);
        size_t othersize = 1;
        for (size_t i = 0; i < ndim - 1; i++) {
            othersize *= input_desc->dim(i);
        }
        size_t feature_size = input_desc->dim(ndim - 1);

        bool bias_exist = bias_desc != nullptr;
        CHECK_OR_RETURN(
            (!bias_exist) || (bias_desc->ndim() == 1 && bias_desc->dim(0) == feature_size),
            INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(
            (weight_desc->ndim() == 1) && (weight_desc->dim(0) == feature_size),
            INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(
            input_std_deviation_desc->ndim() == ndim - 1,
            INFINI_STATUS_BAD_TENSOR_SHAPE);
        for (size_t i = 0; i < ndim - 1; i++) {
            CHECK_OR_RETURN(
                input_std_deviation_desc->dim(i) == input_desc->dim(i),
                INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        return utils::Result<LayerNormInfo>(LayerNormInfo{
            output_desc->dtype(),
            ndim,
            input_desc->shape(),
            normalized_size,
            othersize,
            output_desc->strides(),
            input_standardization_desc->strides(),
            input_std_deviation_desc->strides(),
            input_desc->strides(),
            weight_desc->strides(),
            bias_exist ? bias_desc->strides() : std::vector<ptrdiff_t>(),
            eps,
            bias_exist});
    }
};
} // namespace op::layer_norm

#endif //  __LAYER_NORM_INFO_H__
