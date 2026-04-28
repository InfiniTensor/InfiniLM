#ifndef __HINGE_EMBEDDING_LOSS_CPU_H__
#define __HINGE_EMBEDDING_LOSS_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <vector>

namespace op::hinge_embedding_loss::cpu {

enum class Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2
};

struct HingeEmbeddingLossInfo {
    size_t ndim;
    std::vector<size_t> shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> target_strides;
    std::vector<ptrdiff_t> y_strides;
    size_t input_size;
    double margin;
    Reduction reduction;

    static utils::Result<HingeEmbeddingLossInfo> create(
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        infiniopTensorDescriptor_t y_desc,
        double margin,
        int reduction);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    HingeEmbeddingLossInfo _info;

    Descriptor(infiniDtype_t dtype, HingeEmbeddingLossInfo info,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _dtype(dtype),
          _info(std::move(info)) {}

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

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *input,
        const void *target,
        void *stream) const;
};

} // namespace op::hinge_embedding_loss::cpu

#endif // __HINGE_EMBEDDING_LOSS_CPU_H__
