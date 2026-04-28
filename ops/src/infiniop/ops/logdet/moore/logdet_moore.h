#ifndef __LOGDET_MOORE_H__
#define __LOGDET_MOORE_H__

#include "../../../operator.h"

namespace op::logdet::moore {

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t matrix_size;
    size_t input_size;
    std::vector<ptrdiff_t> input_strides;

    Descriptor(infiniDtype_t dtype, size_t matrix_size, size_t input_size,
               std::vector<ptrdiff_t> input_strides,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          matrix_size(matrix_size),
          input_size(input_size),
          input_strides(std::move(input_strides)) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc);

    size_t workspaceSize() const {
        const size_t elem_size = (_dtype == INFINI_DTYPE_F32) ? sizeof(float) : sizeof(double);
        return matrix_size * matrix_size * elem_size;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::logdet::moore

#endif // __LOGDET_MOORE_H__
