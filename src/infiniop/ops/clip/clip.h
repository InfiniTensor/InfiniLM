#ifndef __CLIP_H__
#define __CLIP_H__

#include "../../elementwise/elementwise.h"
#include "../../operator.h"

/**
 * @brief Define the Clip descriptor for the ternary operator
 *
 * This macro defines a Descriptor class for the Clip operator that inherits from InfiniopDescriptor.
 * It uses the standard elementwise operation fields and methods for a ternary operator
 * where min_val and max_val are tensors.
 *
 * @param OP The operator name (clip)
 * @param NAMESPACE The namespace (cpu or cuda)
 */
#define CLIP_DESCRIPTOR(OP, NAMESPACE)                                        \
                                                                              \
    namespace op::OP::NAMESPACE {                                             \
    class Descriptor final : public InfiniopDescriptor {                      \
        infiniDtype_t _dtype;                                                 \
        op::elementwise::ElementwiseInfo _info;                               \
        std::unique_ptr<op::elementwise::NAMESPACE::DeviceImpl> _device_info; \
        size_t _workspace_size;                                               \
                                                                              \
    public:                                                                   \
        Descriptor(                                                           \
            infiniDtype_t dtype,                                              \
            op::elementwise::ElementwiseInfo info,                            \
            op::elementwise::NAMESPACE::DeviceImpl *device_info,              \
            size_t workspace_size,                                            \
            infiniDevice_t device_type,                                       \
            int device_id)                                                    \
            : InfiniopDescriptor{device_type, device_id},                     \
              _dtype(dtype),                                                  \
              _info(std::move(info)),                                         \
              _device_info(std::move(device_info)),                           \
              _workspace_size(workspace_size) {}                              \
                                                                              \
        ~Descriptor();                                                        \
                                                                              \
        size_t workspaceSize() const { return _workspace_size; }              \
                                                                              \
        infiniStatus_t calculate(                                             \
            void *workspace, size_t workspace_size,                           \
            void *output,                                                     \
            std::vector<const void *> inputs,                                 \
            void *stream) const;                                              \
    };                                                                        \
    }

#endif // __CLIP_H__
