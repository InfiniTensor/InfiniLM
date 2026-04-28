#ifndef __INDEX_COPY_INFO_H__
#define __INDEX_COPY_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::index_copy {

class IndexCopyInfo {
    IndexCopyInfo() = default;

public:
    int _dtype;
    int _idx_dtype;
    int64_t _dim;

    size_t _outer_size;
    size_t _inner_size;
    size_t _dim_size;
    size_t _index_len;

    IndexCopyInfo(int dtype, int idx_dtype, int64_t dim,
                  size_t outer_size, size_t inner_size, size_t dim_size, size_t index_len)
        : _dtype(dtype), _idx_dtype(idx_dtype), _dim(dim),
          _outer_size(outer_size), _inner_size(inner_size), _dim_size(dim_size), _index_len(index_len) {}

    int dtype() const { return _dtype; }
    int idx_dtype() const { return _idx_dtype; }
    int64_t dim() const { return _dim; }

    size_t outer_size() const { return _outer_size; }
    size_t inner_size() const { return _inner_size; }
    size_t dim_size() const { return _dim_size; }
    size_t index_len() const { return _index_len; }

    static utils::Result<IndexCopyInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t in_desc,
        int64_t dim,
        infiniopTensorDescriptor_t index_desc,
        infiniopTensorDescriptor_t source_desc) {

        int dtype = in_desc->dtype();
        if (out_desc->dtype() != dtype || source_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        int idx_dtype = index_desc->dtype();
        if (idx_dtype != INFINI_DTYPE_I32 && idx_dtype != INFINI_DTYPE_I64) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        int64_t ndim = static_cast<int64_t>(in_desc->ndim());
        if (dim < 0 || dim >= ndim) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (index_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &in_shape = in_desc->shape();

        size_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= in_shape[i];
        }

        size_t inner_size = 1;
        for (int64_t i = dim + 1; i < ndim; ++i) {
            inner_size *= in_shape[i];
        }

        size_t dim_size = in_shape[dim];
        size_t index_len = index_desc->shape()[0];

        if (source_desc->ndim() != in_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &src_shape = source_desc->shape();

        for (int64_t i = 0; i < ndim; ++i) {
            if (i == dim) {
                if (src_shape[i] != index_len) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            } else {
                if (src_shape[i] != in_shape[i]) {
                    return INFINI_STATUS_BAD_TENSOR_SHAPE;
                }
            }
        }

        if (out_desc->ndim() != in_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        const auto &out_shape = out_desc->shape();
        for (int64_t i = 0; i < ndim; ++i) {
            if (out_shape[i] != in_shape[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        return utils::Result<IndexCopyInfo>(IndexCopyInfo{
            dtype,
            idx_dtype,
            dim,
            outer_size,
            inner_size,
            dim_size,
            index_len});
    }
};

} // namespace op::index_copy

#endif // __INDEX_COPY_INFO_H__
