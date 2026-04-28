#ifndef __INNER_INFO_H__
#define __INNER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::inner {

class InnerInfo {
    InnerInfo() = default;

public:
    infiniDtype_t dtype;

    size_t input_ndim;
    size_t other_ndim;
    size_t out_ndim;

    size_t oper_len;
    size_t total_elements;

    std::vector<size_t> out_shape;

    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> other_strides;
    std::vector<ptrdiff_t> out_strides;

    static utils::Result<InnerInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t other_desc) {

        auto dtype = out_desc->dtype();
        if (dtype != input_desc->dtype() || dtype != other_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

        auto input_ndim = input_desc->ndim();
        auto other_ndim = other_desc->ndim();
        auto out_ndim = out_desc->ndim();
        if (out_ndim + 2 != input_ndim + other_ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto input_shape = input_desc->shape();
        auto other_shape = other_desc->shape();
        auto out_shape = out_desc->shape();
        if (input_ndim && other_ndim) {
            if (input_shape[input_ndim - 1] != other_shape[other_ndim - 1]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        size_t out_dim_pos = 0;
        if (input_ndim) {
            for (size_t i = 0; i < input_ndim - 1; i++) {
                if (input_shape[i] != out_shape[out_dim_pos++]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }
        if (other_ndim) {
            for (size_t i = 0; i < other_ndim - 1; i++) {
                if (other_shape[i] != out_shape[out_dim_pos++]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        size_t oper_len = 1;
        if (input_ndim && other_ndim) {
            oper_len = input_shape[input_ndim - 1];
        }

        size_t total_elements = 1;
        for (size_t i = 0; i < out_ndim; i++) {
            total_elements *= out_shape[i];
        }

        auto input_strides = input_desc->strides();
        auto other_strides = other_desc->strides();
        auto out_strides = out_desc->strides();

        return utils::Result<InnerInfo>(InnerInfo{
            dtype,
            input_ndim, other_ndim, out_ndim,
            oper_len, total_elements, out_shape,
            input_strides, other_strides, out_strides});
    }
};

} // namespace op::inner

#endif // __INNER_INFO_H__
