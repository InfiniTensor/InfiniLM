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
    size_t seq_len;
    size_t total_seq_len;

    ptrdiff_t y_stride_b;
    ptrdiff_t y_stride_i;
    ptrdiff_t y_stride_j;

    ptrdiff_t x_stride_b;
    ptrdiff_t x_stride_i;
    ptrdiff_t x_stride_j;

    static utils::Result<CausalSoftmaxInfo> create(infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc) {
        auto dtype = y_desc->dtype();
        if (dtype != x_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32);

        auto shape = y_desc->shape();
        CHECK_SAME_SHAPE(shape, x_desc->shape());

        auto ndim = y_desc->ndim();
        if (ndim != 2 && ndim != 3) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        if (shape[ndim - 1] < shape[ndim - 2]) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        size_t batch_size = 1;
        size_t seq_len = shape[ndim - 2];
        size_t total_seq_len = shape[ndim - 1];
        ptrdiff_t y_stride_b = 0,
                  y_stride_i = y_desc->stride(ndim - 2),
                  y_stride_j = y_desc->stride(ndim - 1);
        ptrdiff_t x_stride_b = 0,
                  x_stride_i = x_desc->stride(ndim - 2),
                  x_stride_j = x_desc->stride(ndim - 1);

        if (ndim == 3) {
            y_stride_b = y_desc->stride(0);
            x_stride_b = x_desc->stride(0);
            batch_size = shape[0];
        }

        return utils::Result<CausalSoftmaxInfo>(CausalSoftmaxInfo{
            dtype,
            batch_size,
            seq_len,
            total_seq_len,
            y_stride_b,
            y_stride_i,
            y_stride_j,
            x_stride_b,
            x_stride_i,
            x_stride_j});
    }
};

} // namespace op::causal_softmax

#endif // __CAUSAL_SOFTMAX_INFO_H__
