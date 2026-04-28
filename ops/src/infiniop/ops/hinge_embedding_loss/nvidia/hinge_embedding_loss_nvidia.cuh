#ifndef __HINGE_EMBEDDING_LOSS_NVIDIA_H__
#define __HINGE_EMBEDDING_LOSS_NVIDIA_H__

#include "../../../operator.h"
#include <vector>

namespace op::hinge_embedding_loss::nvidia {

enum class Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    size_t ndim;
    size_t input_size;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> target_strides;
    std::vector<ptrdiff_t> output_strides;
    bool input_contiguous;
    bool target_contiguous;
    bool output_contiguous;
    double margin;
    Reduction reduction;

    Descriptor(infiniDtype_t dtype, size_t ndim, size_t input_size,
               std::vector<size_t> shape,
               std::vector<ptrdiff_t> input_strides,
               std::vector<ptrdiff_t> target_strides,
               std::vector<ptrdiff_t> output_strides,
               bool input_contiguous,
               bool target_contiguous,
               bool output_contiguous,
               double margin, Reduction reduction,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          ndim(ndim),
          input_size(input_size),
          shape(std::move(shape)),
          input_strides(std::move(input_strides)),
          target_strides(std::move(target_strides)),
          output_strides(std::move(output_strides)),
          input_contiguous(input_contiguous),
          target_contiguous(target_contiguous),
          output_contiguous(output_contiguous),
          margin(margin),
          reduction(reduction) {}

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        double margin,
        int reduction);

    size_t workspaceSize() const {
        // For reduction != NONE, reserve 8 bytes for an aligned accumulation scalar.
        const size_t acc_bytes = (reduction == Reduction::NONE) ? 0 : 8;
        const size_t shape_bytes = ndim * sizeof(size_t);
        const size_t stride_bytes = ndim * sizeof(ptrdiff_t);
        const size_t in_target_bytes = 2 * stride_bytes;
        const size_t out_stride_bytes = (reduction == Reduction::NONE) ? stride_bytes : 0;
        return acc_bytes + shape_bytes + in_target_bytes + out_stride_bytes;
    }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *input,
        const void *target,
        void *stream) const;
};

} // namespace op::hinge_embedding_loss::nvidia

#endif // __HINGE_EMBEDDING_LOSS_NVIDIA_H__
