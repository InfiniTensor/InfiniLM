#ifndef __ADD_RMS_NORM_INFO_H__
#define __ADD_RMS_NORM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::add_rms_norm {

class AddRMSNormInfo {
    AddRMSNormInfo() = default;

public:
    infiniDtype_t wtype;
    infiniDtype_t atype;
    float epsilon;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> residual_out_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;
    bool has_residual_out;

    size_t ndim() const { return shape.size(); }
    size_t dim() const { return shape[ndim() - 1]; }

    static utils::Result<AddRMSNormInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t residual_out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t weight_desc,
        float epsilon) {

        auto atype = y_desc->dtype();
        auto wtype = weight_desc->dtype();

        // Check that all input tensors have the same dtype
        if (a_desc->dtype() != atype || b_desc->dtype() != atype) {
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
        const size_t a_ndim = a_desc->ndim();
        const size_t b_ndim = b_desc->ndim();
        const size_t w_ndim = weight_desc->ndim();

        if (y_ndim != a_ndim || y_ndim != b_ndim || w_ndim != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t batch = 1;
        size_t nhead = 1;
        size_t dim = 0;

        if (y_ndim == 2) {
            batch = y_desc->dim(0);
            dim = y_desc->dim(1);

            if (a_desc->dim(0) != batch || a_desc->dim(1) != dim || b_desc->dim(0) != batch || b_desc->dim(1) != dim || weight_desc->dim(0) != dim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (y_ndim == 3) {
            batch = y_desc->dim(0);
            nhead = y_desc->dim(1);
            dim = y_desc->dim(2);

            if (a_desc->dim(0) != batch || a_desc->dim(1) != nhead || a_desc->dim(2) != dim || b_desc->dim(0) != batch || b_desc->dim(1) != nhead || b_desc->dim(2) != dim || weight_desc->dim(0) != dim) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Check contiguity of the last dimension
        if (y_desc->stride(y_ndim - 1) != 1 || a_desc->stride(a_ndim - 1) != 1 || b_desc->stride(b_ndim - 1) != 1 || weight_desc->stride(w_ndim - 1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        // residual_out_desc is required (always needed for fused operator)
        if (residual_out_desc == nullptr) {
            return INFINI_STATUS_BAD_PARAM;
        }

        const size_t residual_out_ndim = residual_out_desc->ndim();
        if (residual_out_ndim != y_ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (residual_out_desc->dtype() != atype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        // Check shape matches
        for (size_t i = 0; i < y_ndim; i++) {
            if (residual_out_desc->dim(i) != y_desc->dim(i)) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }
        if (residual_out_desc->stride(residual_out_ndim - 1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        AddRMSNormInfo info;
        info.wtype = wtype;
        info.atype = atype;
        info.epsilon = epsilon;
        info.shape = y_desc->shape();
        info.y_strides = y_desc->strides();
        info.a_strides = a_desc->strides();
        info.b_strides = b_desc->strides();
        info.has_residual_out = true; // Always true now
        info.residual_out_strides = residual_out_desc->strides();
        return utils::Result<AddRMSNormInfo>(info);
    }
};

} // namespace op::add_rms_norm

#endif // __ADD_RMS_NORM_INFO_H__
