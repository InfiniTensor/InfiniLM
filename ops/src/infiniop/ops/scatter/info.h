#ifndef __SCATTER_INFO_H__
#define __SCATTER_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::scatter {

class ScatterInfo {
    ScatterInfo() = default;

public:
    int _dtype;
    int _idx_dtype;
    int _axis;
    int _reduction;
    size_t _ndim;

    int dtype() const { return _dtype; }
    int idx_dtype() const { return _idx_dtype; }
    int axis() const { return _axis; }
    int reduction() const { return _reduction; }
    size_t ndim() const { return _ndim; }

    ScatterInfo(int dtype, int idx_dtype, int axis, int reduction, size_t ndim)
        : _dtype(dtype), _idx_dtype(idx_dtype), _axis(axis), _reduction(reduction), _ndim(ndim) {}

    static utils::Result<ScatterInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t indices_desc,
        infiniopTensorDescriptor_t updates_desc,
        int axis,
        int reduction) {

        size_t ndim = input_desc->ndim();
        if (out_desc->ndim() != ndim || indices_desc->ndim() != ndim || updates_desc->ndim() != ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        int canonical_axis = axis;
        if (canonical_axis < 0) {
            canonical_axis += static_cast<int>(ndim);
        }
        if (canonical_axis < 0 || canonical_axis >= static_cast<int>(ndim)) {
            return INFINI_STATUS_BAD_PARAM;
        }

        const auto &in_shape = input_desc->shape();
        const auto &out_shape = out_desc->shape();
        for (size_t i = 0; i < ndim; ++i) {
            if (in_shape[i] != out_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        const auto &idx_shape = indices_desc->shape();
        const auto &upd_shape = updates_desc->shape();
        for (size_t i = 0; i < ndim; ++i) {
            if (idx_shape[i] != upd_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        for (size_t i = 0; i < ndim; ++i) {
            if (idx_shape[i] > in_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        if (input_desc->dtype() != updates_desc->dtype() || input_desc->dtype() != out_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (indices_desc->dtype() != INFINI_DTYPE_I32 && indices_desc->dtype() != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (reduction < 0 || reduction > 2) {
            return INFINI_STATUS_BAD_PARAM;
        }

        return utils::Result<ScatterInfo>(ScatterInfo{
            input_desc->dtype(),
            indices_desc->dtype(),
            canonical_axis,
            reduction,
            ndim});
    }
};

} // namespace op::scatter

#endif // __SCATTER_INFO_H__
