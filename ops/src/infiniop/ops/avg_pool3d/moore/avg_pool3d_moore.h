#ifndef __AVG_POOL3D_MOORE_H__
#define __AVG_POOL3D_MOORE_H__

#include "../../../operator.h"
#include "../../../tensor.h"
#include <memory>

namespace op::avg_pool3d::moore {

class Descriptor final : public InfiniopDescriptor {
    struct Opaque;
    std::unique_ptr<Opaque> _opaque;
    infiniDtype_t _dtype;

    Descriptor(infiniDtype_t dtype, std::unique_ptr<Opaque> opaque,
               infiniDevice_t device_type, int device_id);

public:
    ~Descriptor();

    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        void *kernel_size,
        void *stride,
        void *padding);

    size_t workspaceSize() const;

    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *y,
        const void *x,
        void *stream) const;
};

} // namespace op::avg_pool3d::moore

#endif // __AVG_POOL3D_MOORE_H__
