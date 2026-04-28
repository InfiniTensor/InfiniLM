#ifndef __MULTI_MARGIN_LOSS_H__
#define __MULTI_MARGIN_LOSS_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 MultiMarginLossInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::multi_margin_loss::NAMESPACE {                 \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        MultiMarginLossInfo _info;                               \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            MultiMarginLossInfo info,                            \
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
            infiniopTensorDescriptor_t weight_desc,              \
            int p,                                               \
            float margin,                                        \
            int reduction);                                      \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *input,                                   \
            const void *target,                                  \
            const void *weight,                                  \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __MULTI_MARGIN_LOSS_H__
