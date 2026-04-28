#ifndef __GAUSSIAN_NLL_LOSS_METAX_H__
#define __GAUSSIAN_NLL_LOSS_METAX_H__

#include "../../../operator.h"

namespace op::gaussian_nll_loss::metax {

enum class Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t input_size;
    size_t ndim;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> target_strides;
    std::vector<ptrdiff_t> var_strides;
    int full;
    double eps;
    Reduction reduction;
    void *reduce_buffer;

    Descriptor(infiniDtype_t dtype,
               size_t input_size,
               size_t ndim,
               std::vector<size_t> shape,
               std::vector<ptrdiff_t> y_strides,
               std::vector<ptrdiff_t> input_strides,
               std::vector<ptrdiff_t> target_strides,
               std::vector<ptrdiff_t> var_strides,
               int full,
               double eps,
               Reduction reduction,
               void *reduce_buffer,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          input_size(input_size),
          ndim(ndim),
          shape(std::move(shape)),
          y_strides(std::move(y_strides)),
          input_strides(std::move(input_strides)),
          target_strides(std::move(target_strides)),
          var_strides(std::move(var_strides)),
          full(full),
          eps(eps),
          reduction(reduction),
          reduce_buffer(reduce_buffer) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        infiniopTensorDescriptor_t var_desc,
        int full,
        double eps,
        int reduction);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *input,
        const void *target,
        const void *var,
        void *stream) const;
};

} // namespace op::gaussian_nll_loss::metax

#endif // __GAUSSIAN_NLL_LOSS_METAX_H__
