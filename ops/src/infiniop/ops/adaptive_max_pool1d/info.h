#ifndef __ADAPATIVE_MAX_POOL1D_H__
#define __ADAPATIVE_MAX_POOL1D_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::adaptive_max_pool1d {

class AdaptiveMaxPool1dInfo {
    AdaptiveMaxPool1dInfo() = default;

public:
    infiniDtype_t atype;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;
    size_t input_size;
    size_t output_size;
    size_t ndim() const { return shape.size(); }
    size_t input_length() const { return input_size; }
    size_t output_length() const { return output_size; }

    static utils::Result<AdaptiveMaxPool1dInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        size_t output_size) {

        auto atype = y_desc->dtype();
        if (x_desc->dtype() != atype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (atype != INFINI_DTYPE_F16 && atype != INFINI_DTYPE_BF16 && atype != INFINI_DTYPE_F32 && atype != INFINI_DTYPE_F64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        const size_t y_ndim = y_desc->ndim();
        const size_t x_ndim = x_desc->ndim();

        if (y_ndim != x_ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (size_t i = 0; i < y_ndim - 1; ++i) {
            if (x_desc->dim(i) != y_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        if (y_desc->dim(y_ndim - 1) != output_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<AdaptiveMaxPool1dInfo>(AdaptiveMaxPool1dInfo{
            atype,
            y_desc->shape(),
            y_desc->strides(),
            x_desc->strides(),
            x_desc->dim(x_ndim - 1),
            output_size});
    }
};
} // namespace op::adaptive_max_pool1d

#endif // __ADAPATIVE_MAX_POOL1D_H__
