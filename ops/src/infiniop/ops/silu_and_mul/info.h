#ifndef __SILU_AND_MUL_INFO_H__
#define __SILU_AND_MUL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::silu_and_mul {

class SiluAndMulInfo {
    SiluAndMulInfo() = default;

public:
    infiniDtype_t dtype;
    size_t batch_size;
    size_t out_hidden_dim;

    static utils::Result<SiluAndMulInfo> create(infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc) {
        auto dtype = y_desc->dtype();

        auto x_shape = x_desc->shape();
        auto y_shape = y_desc->shape();
        auto ndim = x_desc->ndim();

        if (ndim != y_desc->ndim()) {
            return INFINI_STATUS_BAD_PARAM;
        }

        if (x_shape[ndim - 1] != 2 * y_shape[ndim - 1]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t batch = 1;
        for (int i = 0; i < (int)ndim - 1; ++i) {
            if (x_shape[i] != y_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
            batch *= y_shape[i];
        }

        return utils::Result<SiluAndMulInfo>(SiluAndMulInfo{
            dtype,
            batch,
            y_shape[ndim - 1]});
    }

private:
    SiluAndMulInfo(infiniDtype_t dtype, size_t batch, size_t hidden)
        : dtype(dtype), batch_size(batch), out_hidden_dim(hidden) {}
};

} // namespace op::silu_and_mul

#endif // __SILU_AND_MUL_INFO_H__
