#ifndef __LOG_SOFTMAX_INFO_H__
#define __LOG_SOFTMAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::log_softmax {

class LogSoftmaxInfo {
    LogSoftmaxInfo() = default;

public:
    int _dtype;
    int _dim;

    size_t _dim_size;
    size_t _outer_size;
    size_t _inner_size;

    int dtype() const { return _dtype; }
    int dim() const { return _dim; }
    size_t dim_size() const { return _dim_size; }
    size_t outer_size() const { return _outer_size; }
    size_t inner_size() const { return _inner_size; }

    LogSoftmaxInfo(int dtype, int dim, size_t dim_size, size_t outer_size, size_t inner_size)
        : _dtype(dtype), _dim(dim),
          _dim_size(dim_size), _outer_size(outer_size), _inner_size(inner_size) {}

    static utils::Result<LogSoftmaxInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int dim) {

        int ndim = int(input_desc->ndim());

        if (dim < 0) {
            dim += ndim;
        }
        if (dim < 0 || dim >= ndim) {
            return INFINI_STATUS_BAD_PARAM;
        }

        size_t dim_size = input_desc->shape()[dim];

        size_t outer_size = 1;
        for (int i = 0; i < dim; ++i) {
            outer_size *= input_desc->shape()[i];
        }

        size_t inner_size = 1;
        for (int i = dim + 1; i < ndim; ++i) {
            inner_size *= input_desc->shape()[i];
        }

        // Validate Shape: LogSoftmax requires input and output shapes to be identical
        if (output_desc->ndim() != input_desc->ndim()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        for (int i = 0; i < ndim; ++i) {
            if (output_desc->shape()[i] != input_desc->shape()[i]) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        }

        // Validate Dtype
        if (output_desc->dtype() != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        return utils::Result<LogSoftmaxInfo>(LogSoftmaxInfo{
            input_desc->dtype(),
            dim,
            dim_size,
            outer_size,
            inner_size});
    }
};

} // namespace op::log_softmax

#endif // __LOG_SOFTMAX_INFO_H__
