#ifndef __SCATTER_H__
#define __SCATTER_H__

#include "../../operator.h"
#include "info.h"

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::scatter::NAMESPACE {                           \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        ScatterInfo _info;                                       \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            ScatterInfo info,                                    \
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
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t output,                   \
            infiniopTensorDescriptor_t input,                    \
            infiniopTensorDescriptor_t indices,                  \
            infiniopTensorDescriptor_t updates,                  \
            int axis,                                            \
            int reduction);                                      \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *input,                                   \
            const void *indices,                                 \
            const void *updates,                                 \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __SCATTER_H__
