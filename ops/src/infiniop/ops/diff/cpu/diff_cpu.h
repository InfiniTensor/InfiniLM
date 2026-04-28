#ifndef __DIFF_CPU_H__
#define __DIFF_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <vector>

namespace op::diff::cpu {

struct DiffInfo {
    size_t ndim;
    int dim;
    int n;
    std::vector<size_t> input_shape;
    std::vector<size_t> output_shape;
    std::vector<ptrdiff_t> input_strides;
    std::vector<ptrdiff_t> output_strides;
    size_t input_size;
    size_t output_size;

    static utils::Result<DiffInfo> create(
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t y_desc,
        int dim,
        int n);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    DiffInfo _info;

    Descriptor(infiniDtype_t dtype, DiffInfo info,
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
        infiniopTensorDescriptor_t x_desc,
        int dim,
        int n);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::diff::cpu

#endif // __DIFF_CPU_H__
