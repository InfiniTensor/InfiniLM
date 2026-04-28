#ifndef __DOT_METAX_H__
#define __DOT_METAX_H__

#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::dot::metax {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t _n;
    ptrdiff_t _a_stride;
    ptrdiff_t _b_stride;

    Descriptor(infiniDtype_t dtype, size_t n, ptrdiff_t a_stride, ptrdiff_t b_stride,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _n(n),
          _a_stride(a_stride),
          _b_stride(b_stride) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *a,
        const void *b,
        void *stream) const;
};

} // namespace op::dot::metax

#endif // __DOT_METAX_H__
