#ifndef __RMS_NORM_INFO_H__
#define __RMS_NORM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::rms_norm {

class RMSNormInfo {
    RMSNormInfo() = default;

public:
    infiniDtype_t wtype;
    infiniDtype_t atype;
    float epsilon;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;

    size_t ndim() const { return shape.size(); }
    size_t dim() const { return shape[ndim() - 1]; }

    static utils::Result<RMSNormInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        float epsilon) {

        auto atype = y_desc->dtype();
        auto wtype = w_desc->dtype();
        if (x_desc->dtype() != atype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (atype == INFINI_DTYPE_F16 || atype == INFINI_DTYPE_BF16) {
            // For half-precision types (FP16/BF16), weights can be the same half-precision type or FP32
            if (wtype != atype && wtype != INFINI_DTYPE_F32 && wtype != INFINI_DTYPE_BF16 && wtype != INFINI_DTYPE_F16) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else if (atype == INFINI_DTYPE_F32 || atype == INFINI_DTYPE_F64) {
            // For FP32/FP64, activations and weights must be of the same type
            if (atype != wtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        const size_t y_ndim = y_desc->ndim();
        const size_t x_ndim = x_desc->ndim();
        const size_t w_ndim = w_desc->ndim();

        if (y_ndim != x_ndim || w_ndim != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t batch = 1;
        size_t nhead = 1;
        size_t dim = 0;

        if (y_ndim == 2) {
            batch = y_desc->dim(0);
            dim = y_desc->dim(1);

            if (x_desc->dim(0) != batch || x_desc->dim(1) != dim || w_desc->dim(0) != dim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (y_ndim == 3) {
            batch = y_desc->dim(0);
            nhead = y_desc->dim(1);
            dim = y_desc->dim(2);

            if (x_desc->dim(0) != batch || x_desc->dim(1) != nhead || x_desc->dim(2) != dim || w_desc->dim(0) != dim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check contiguity of the last dimension
        if (y_desc->stride(y_ndim - 1) != 1 || x_desc->stride(x_ndim - 1) != 1 || w_desc->stride(w_ndim - 1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<RMSNormInfo>(RMSNormInfo{
            wtype,
            atype,
            epsilon,
            y_desc->shape(),
            y_desc->strides(),
            x_desc->strides(),
        });
    }
};

} // namespace op::rms_norm

#endif // __RMS_NORM_INFO_H__
