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
        if (atype == INFINI_DTYPE_F16) {
            if (wtype != INFINI_DTYPE_F16 && wtype != INFINI_DTYPE_F32) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else if (atype == INFINI_DTYPE_F32 || atype == INFINI_DTYPE_F64) {
            if (atype != wtype) {
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (y_desc->ndim() != 2 || x_desc->ndim() != 2 || w_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t batch = y_desc->shape()[0];
        size_t dim = y_desc->shape()[1];
        if (x_desc->shape()[0] != batch || x_desc->shape()[1] != dim || w_desc->shape()[0] != dim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (w_desc->stride(0) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        if (x_desc->stride(1) != 1 || y_desc->stride(1) != 1) {
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
