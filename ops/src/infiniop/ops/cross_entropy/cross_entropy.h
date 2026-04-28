#ifndef CROSS_ENTROPY_H
#define CROSS_ENTROPY_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                 \
    namespace op::cross_entropy::NAMESPACE {                                  \
    class Descriptor final : public InfiniopDescriptor {                      \
        struct Opaque;                                                        \
        Opaque *_opaque;                                                      \
        CrossEntropyInfo _info;                                               \
        size_t _workspace_size;                                               \
                                                                              \
        Descriptor(Opaque *opaque,                                            \
                   CrossEntropyInfo info,                                     \
                   size_t workspace_size,                                     \
                   infiniDevice_t device_type,                                \
                   int device_id)                                             \
            : InfiniopDescriptor{device_type, device_id},                     \
              _opaque(opaque),                                                \
              _info(info),                                                    \
              _workspace_size(workspace_size) {}                              \
                                                                              \
    public:                                                                   \
        ~Descriptor();                                                        \
        size_t workspaceSize() const { return _workspace_size; }              \
        static infiniStatus_t create(infiniopHandle_t handle,                 \
                                     Descriptor **desc_ptr,                   \
                                     infiniopTensorDescriptor_t y_desc,       \
                                     infiniopTensorDescriptor_t x_desc,       \
                                     infiniopTensorDescriptor_t target_desc); \
        infiniStatus_t calculate(void *workspace,                             \
                                 size_t workspace_size,                       \
                                 void *y,                                     \
                                 const void *x,                               \
                                 const void *target,                          \
                                 void *stream) const;                         \
    };                                                                        \
    }

#endif
