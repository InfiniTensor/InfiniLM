#ifndef __DIST_CPU_H__
#define __DIST_CPU_H__

#include "../../../devices/cpu/common_cpu.h"
#include "../../../operator.h"
#include <cmath>
#include <vector>

namespace op::dist::cpu {

struct DistInfo {
    size_t input_size;
    double p;
    std::vector<ptrdiff_t> x1_strides;
    std::vector<ptrdiff_t> x2_strides;
    std::vector<size_t> shape;
    size_t ndim;

    static utils::Result<DistInfo> create(
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t x2_desc,
        infiniopTensorDescriptor_t y_desc,
        double p);
};

class Descriptor final : public InfiniopDescriptor {
    infiniDtype_t _dtype;
    DistInfo _info;

    Descriptor(infiniDtype_t dtype, DistInfo info,
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
        infiniopTensorDescriptor_t x1_desc,
        infiniopTensorDescriptor_t x2_desc,
        double p);

    size_t workspaceSize() const { return 0; }

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x1,
        const void *x2,
        void *stream) const;
};

} // namespace op::dist::cpu

#endif // __DIST_CPU_H__
