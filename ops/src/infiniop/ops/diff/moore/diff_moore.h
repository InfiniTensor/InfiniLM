#ifndef __DIFF_MOORE_H__
#define __DIFF_MOORE_H__

#include "../../../operator.h"

namespace op::diff::moore {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t _ndim;
    int _dim;
    int _n;
    std::vector<size_t> _input_shape;
    std::vector<size_t> _output_shape;
    std::vector<ptrdiff_t> _input_strides;
    std::vector<ptrdiff_t> _output_strides;
    size_t _input_size;
    size_t _output_size;

    Descriptor(infiniDtype_t dtype, size_t ndim, int dim, int n,
               std::vector<size_t> input_shape, std::vector<size_t> output_shape,
               std::vector<ptrdiff_t> input_strides, std::vector<ptrdiff_t> output_strides,
               size_t input_size, size_t output_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _ndim(ndim),
          _dim(dim),
          _n(n),
          _input_shape(std::move(input_shape)),
          _output_shape(std::move(output_shape)),
          _input_strides(std::move(input_strides)),
          _output_strides(std::move(output_strides)),
          _input_size(input_size),
          _output_size(output_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int dim,
        int n);

    size_t workspaceSize() const {
        if (_n <= 1) {
            return 0;
        }
        const size_t dim_size = _input_shape[static_cast<size_t>(_dim)];
        const size_t outer = _input_size / dim_size;
        const size_t max_intermediate = outer * (dim_size - 1);
        return 2 * max_intermediate * infiniSizeOf(_dtype);
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};
} // namespace op::diff::moore

#endif // __DIFF_MOORE_H__
