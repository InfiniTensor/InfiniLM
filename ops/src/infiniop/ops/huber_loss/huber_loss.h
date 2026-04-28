#ifndef __HUBER_LOSS_H__
#define __HUBER_LOSS_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 HuberLossInfo 定义
#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::huber_loss::NAMESPACE {                        \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        HuberLossInfo _info;                                     \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            HuberLossInfo info,                                  \
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
            infiniopTensorDescriptor_t out_desc,                 \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t target_desc,              \
            float delta,                                         \
            int reduction);                                      \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *input,                                   \
            const void *target,                                  \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __HUBER_LOSS_H__
