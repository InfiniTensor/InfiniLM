#ifndef __KRON_METAX_H__
#define __KRON_METAX_H__

#include "../../../operator.h"
#include <cstddef>
#include <vector>

namespace op::kron::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t ndim;
    std::vector<size_t> a_shape;
    std::vector<size_t> b_shape;
    std::vector<size_t> y_shape;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;
    std::vector<ptrdiff_t> y_strides;
    size_t a_size;
    size_t b_size;
    size_t y_size;

    Descriptor(infiniDtype_t dtype, size_t ndim,
               std::vector<size_t> a_shape, std::vector<size_t> b_shape, std::vector<size_t> y_shape,
               std::vector<ptrdiff_t> a_strides,
               std::vector<ptrdiff_t> b_strides,
               std::vector<ptrdiff_t> y_strides,
               size_t a_size, size_t b_size, size_t y_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          ndim(ndim),
          a_shape(std::move(a_shape)),
          b_shape(std::move(b_shape)),
          y_shape(std::move(y_shape)),
          a_strides(std::move(a_strides)),
          b_strides(std::move(b_strides)),
          y_strides(std::move(y_strides)),
          a_size(a_size),
          b_size(b_size),
          y_size(y_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc);

    size_t workspaceSize() const {
        return (ndim * 3) * sizeof(size_t) + (ndim * 3) * sizeof(ptrdiff_t);
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *a,
        const void *b,
        void *stream) const;
};

} // namespace op::kron::metax

#endif // __KRON_METAX_H__
