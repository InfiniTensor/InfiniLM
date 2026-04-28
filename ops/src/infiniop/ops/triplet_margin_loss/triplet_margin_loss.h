#ifndef __TRIPLET_MARGIN_LOSS_H__
#define __TRIPLET_MARGIN_LOSS_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 TripletMarginLossInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::triplet_margin_loss::NAMESPACE {               \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        TripletMarginLossInfo _info;                             \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            TripletMarginLossInfo info,                          \
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
            infiniopTensorDescriptor_t output_desc,              \
            infiniopTensorDescriptor_t anchor_desc,              \
            infiniopTensorDescriptor_t positive_desc,            \
            infiniopTensorDescriptor_t negative_desc,            \
            float margin,                                        \
            int p,                                               \
            float eps,                                           \
            int swap,                                            \
            int reduction);                                      \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *anchor,                                  \
            const void *positive,                                \
            const void *negative,                                \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __TRIPLET_MARGIN_LOSS_H__
