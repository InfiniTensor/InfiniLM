#ifndef __KTHVALUE_INFO_H__
#define __KTHVALUE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::kthvalue {

class KthvalueInfo {
    KthvalueInfo() = default;

public:
    int _dtype;
    int _indices_dtype;
    int _k;
    int _dim;
    bool _keepdim;

    size_t _dim_size;
    size_t _outer_size;
    size_t _inner_size;

    int dtype() const { return _dtype; }
    int indices_dtype() const { return _indices_dtype; }
    int k() const { return _k; }
    int dim() const { return _dim; }
    bool keepdim() const { return _keepdim; }
    size_t dim_size() const { return _dim_size; }
    size_t outer_size() const { return _outer_size; }
    size_t inner_size() const { return _inner_size; }

    KthvalueInfo(int dtype, int indices_dtype, int k, int dim, bool keepdim,
                 size_t dim_size, size_t outer_size, size_t inner_size)
        : _dtype(dtype), _indices_dtype(indices_dtype), _k(k), _dim(dim), _keepdim(keepdim),
          _dim_size(dim_size), _outer_size(outer_size), _inner_size(inner_size) {}

    static utils::Result<KthvalueInfo> create(
        infiniopTensorDescriptor_t values_desc,
        infiniopTensorDescriptor_t indices_desc,
        infiniopTensorDescriptor_t input_desc,
        int k,
        int dim,
        int keepdim) {

        int ndim = int(input_desc->ndim());

        if (dim < 0) {
            dim += ndim;
        }
        if (dim < 0 || dim >= ndim) {
            return INFINI_STATUS_BAD_PARAM;
        }

        size_t dim_size = input_desc->shape()[dim];
        if (k <= 0 || k > static_cast<int>(dim_size)) {
            return INFINI_STATUS_BAD_PARAM;
        }

        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= input_desc->shape()[i];
        }

        size_t inner_size = 1;
        for (int i = dim + 1; i < ndim; ++i) {
            inner_size *= input_desc->shape()[i];
        }

        if (values_desc->ndim() != indices_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // 修复 1: 使用 size_t 避免有符号/无符号比较警告
        size_t expected_out_ndim = keepdim ? ndim : ndim - 1;
        if (expected_out_ndim == 0) {
            expected_out_ndim = 1;
        }

        if (values_desc->ndim() != expected_out_ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        int out_idx = 0;
        for (int i = 0; i < ndim; ++i) {
            if (keepdim) {
                size_t expected_size = (i == dim) ? 1 : input_desc->shape()[i];
                if (values_desc->shape()[i] != expected_size || indices_desc->shape()[i] != expected_size) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            } else {
                if (i == dim) {
                    continue;
                }
                if (values_desc->shape()[out_idx] != input_desc->shape()[i] || indices_desc->shape()[out_idx] != input_desc->shape()[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
                out_idx++;
            }
        }

        if (values_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (indices_desc->dtype() != INFINI_DTYPE_I64 && indices_desc->dtype() != INFINI_DTYPE_I32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return utils::Result<KthvalueInfo>(KthvalueInfo{
            input_desc->dtype(),
            indices_desc->dtype(),
            k,
            dim,
            static_cast<bool>(keepdim),
            dim_size,
            outer_size,
            inner_size});
    }
};

} // namespace op::kthvalue

#endif
