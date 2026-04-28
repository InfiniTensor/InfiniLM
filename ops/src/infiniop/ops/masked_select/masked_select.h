#ifndef MASKED_SELECT_H
#define MASKED_SELECT_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::masked_select::NAMESPACE {                     \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        MaskedSelectInfo _info;                                  \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            MaskedSelectInfo info,                               \
            size_t workspace_size,                               \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _info(info),                                       \
              _workspace_size(workspace_size) {}                 \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t mask_desc);               \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            const void *input,                                   \
            const bool *mask,                                    \
            void **data_ptr,                                     \
            size_t *dlen_ptr,                                    \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // MASKED_SELECT_H
