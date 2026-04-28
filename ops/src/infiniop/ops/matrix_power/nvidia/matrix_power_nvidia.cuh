#ifndef __MATRIX_POWER_NVIDIA_H__
#define __MATRIX_POWER_NVIDIA_H__

#include "../../../operator.h"
#include <cstddef>

namespace op::matrix_power::nvidia {

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    Opaque *_opaque;
    infiniDtype_t _dtype;
    size_t matrix_size;
    size_t n;
    size_t input_size;
    size_t output_size;
    size_t workspace_size;
    ptrdiff_t x_stride_0;
    ptrdiff_t x_stride_1;
    ptrdiff_t y_stride_0;
    ptrdiff_t y_stride_1;
    bool x_contiguous;
    bool y_contiguous;

    Descriptor(infiniDtype_t dtype, size_t matrix_size, size_t n,
               size_t input_size, size_t output_size,
               size_t workspace_size,
               ptrdiff_t x_stride_0, ptrdiff_t x_stride_1,
               ptrdiff_t y_stride_0, ptrdiff_t y_stride_1,
               bool x_contiguous, bool y_contiguous,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _opaque(nullptr),
          _dtype(dtype),
          matrix_size(matrix_size),
          n(n),
          input_size(input_size),
          output_size(output_size),
          workspace_size(workspace_size),
          x_stride_0(x_stride_0),
          x_stride_1(x_stride_1),
          y_stride_0(y_stride_0),
          y_stride_1(y_stride_1),
          x_contiguous(x_contiguous),
          y_contiguous(y_contiguous) {}

public:
    ~Descriptor();

    friend struct Opaque;

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int n);

    size_t workspaceSize() const { return workspace_size; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::matrix_power::nvidia

#endif // __MATRIX_POWER_NVIDIA_H__
