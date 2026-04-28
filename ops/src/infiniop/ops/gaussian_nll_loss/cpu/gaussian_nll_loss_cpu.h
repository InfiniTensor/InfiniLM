#ifndef __GAUSSIAN_NLL_LOSS_CPU_H__
#define __GAUSSIAN_NLL_LOSS_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include "../../../tensor.h"
#include <vector>

namespace op::gaussian_nll_loss::cpu {

enum class Reduction {
    NONE = 0,
    MEAN = 1,
    SUM = 2
};

struct GaussianNllLossInfo {
    size_t input_size;
    int full;
    double eps;
    Reduction reduction;

    static utils::Result<GaussianNllLossInfo> create(
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t target_desc,
        infiniopTensorDescriptor_t var_desc,
        infiniopTensorDescriptor_t y_desc,
        int full,
        double eps,
        int reduction);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    GaussianNllLossInfo _info;

    Descriptor(infiniDtype_t dtype, GaussianNllLossInfo info,
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

} // namespace op::gaussian_nll_loss::cpu

#endif // __GAUSSIAN_NLL_LOSS_CPU_H__
