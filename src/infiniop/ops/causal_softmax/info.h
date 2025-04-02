#ifndef __CAUSAL_SOFTMAX_INFO_H__
#define __CAUSAL_SOFTMAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::causal_softmax {

class CausalSoftmaxInfo {
    CausalSoftmaxInfo() = default;

public:
    infiniDtype_t dtype;
    size_t batch_size;
    ptrdiff_t stride_b;
    size_t seq_len;
    ptrdiff_t stride_i;
    size_t total_seq_len;
    ptrdiff_t stride_j;

    static utils::Result<CausalSoftmaxInfo> create(infiniopTensorDescriptor_t y_desc) {
        auto dtype = y_desc->dtype();
        if (y_desc->dtype() != INFINI_DTYPE_F16 && y_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (y_desc->ndim() != 2 && y_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (y_desc->shape()[y_desc->ndim() - 1] < y_desc->shape()[y_desc->ndim() - 2]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t batch_size = 1;
        ptrdiff_t stride_b = 0;
        size_t seq_len = y_desc->shape()[y_desc->ndim() - 2];
        ptrdiff_t stride_i = y_desc->strides()[y_desc->ndim() - 2];
        size_t total_seq_len = y_desc->shape()[y_desc->ndim() - 1];
        ptrdiff_t stride_j = y_desc->strides()[y_desc->ndim() - 1];
        if (y_desc->ndim() == 3) {
            stride_b = y_desc->strides()[0];
            batch_size = y_desc->shape()[0];
        }

        return utils::Result<CausalSoftmaxInfo>(CausalSoftmaxInfo{
            dtype,
            batch_size,
            stride_b,
            seq_len,
            stride_i,
            total_seq_len,
            stride_j});
    }
};

} // namespace op::causal_softmax

#endif // __CAUSAL_SOFTMAX_INFO_H__
