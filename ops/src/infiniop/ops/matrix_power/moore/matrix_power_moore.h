#ifndef __MATRIX_POWER_MOORE_H__
#define __MATRIX_POWER_MOORE_H__

#include "../../../operator.h"

namespace op::matrix_power::moore {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t matrix_size;
    size_t n;
    size_t input_size;
    size_t output_size;

    Descriptor(infiniDtype_t dtype, size_t matrix_size, size_t n,
               size_t input_size, size_t output_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          matrix_size(matrix_size),
          n(n),
          input_size(input_size),
          output_size(output_size) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        int n);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::matrix_power::moore

#endif // __MATRIX_POWER_MOORE_H__
