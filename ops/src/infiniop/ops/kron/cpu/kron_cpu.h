#ifndef __KRON_CPU_H__
#define __KRON_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <vector>

namespace op::kron::cpu {

struct KronInfo {
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

    static utils::Result<KronInfo> create(
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t y_desc);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    KronInfo _info;

    Descriptor(infiniDtype_t dtype, KronInfo info,
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

} // namespace op::kron::cpu

#endif // __KRON_CPU_H__
