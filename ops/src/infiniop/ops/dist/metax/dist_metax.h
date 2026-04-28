#ifndef __DIST_METAX_H__
#define __DIST_METAX_H__

#include "../../../operator.h"

namespace op::dist::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t _input_size;
    double _p;
    size_t _ndim;
    std::vector<size_t> _shape;
    std::vector<ptrdiff_t> _x1_strides;
    std::vector<ptrdiff_t> _x2_strides;

    Descriptor(infiniDtype_t dtype, size_t input_size, double p,
               size_t ndim, std::vector<size_t> shape,
               std::vector<ptrdiff_t> x1_strides, std::vector<ptrdiff_t> x2_strides,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _input_size(input_size),
          _p(p),
          _ndim(ndim),
          _shape(std::move(shape)),
          _x1_strides(std::move(x1_strides)),
          _x2_strides(std::move(x2_strides)) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t x2_desc,
        double p);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x1,
        const void *x2,
        void *stream) const;
};

} // namespace op::dist::metax

#endif // __DIST_METAX_H__
