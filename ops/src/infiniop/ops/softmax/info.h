#ifndef __SOFTMAX_INFO_H__
#define __SOFTMAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::softmax {

class SoftmaxInfo {
    SoftmaxInfo() = default;

public:
    infiniDtype_t dtype;

    size_t othersize;
    size_t dimsize;

    ptrdiff_t stride;

    static utils::Result<SoftmaxInfo> create(infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, int axis) {
        auto dtype = y_desc->dtype();
        if (dtype != x_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        auto shape = y_desc->shape();
        CHECK_SAME_SHAPE(shape, x_desc->shape());

        auto ndim = y_desc->ndim();

        if (axis < 0) {
            axis += (int)(ndim);
        }
        size_t othersize = 1;
        for (int i = 0; i < (int)ndim; i++) {
            if (i != axis) {
                othersize *= shape[i];
            }
        }
        size_t dimsize = shape[axis];

        ptrdiff_t stride = 1;
        for (int i = ndim - 1; i > axis; i--) {
            stride *= (ptrdiff_t)shape[i];
        }

        return utils::Result<SoftmaxInfo>(SoftmaxInfo{
            dtype,
            othersize,
            dimsize,
            stride});
    }
};

} // namespace op::softmax

#endif // __SOFTMAX_INFO_H__
