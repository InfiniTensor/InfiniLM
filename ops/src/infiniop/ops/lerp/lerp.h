#ifndef __LERP_H__
#define __LERP_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 LerpInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                                                \
    namespace op::lerp::NAMESPACE {                                                          \
    class Descriptor final : public InfiniopDescriptor {                                     \
        struct Opaque;                                                                       \
        Opaque *_opaque;                                                                     \
        LerpInfo _info;                                                                      \
        size_t _workspace_size;                                                              \
                                                                                             \
        Descriptor(                                                                          \
            Opaque *opaque,                                                                  \
            LerpInfo info,                                                                   \
            size_t workspace_size,                                                           \
            infiniDevice_t device_type,                                                      \
            int device_id)                                                                   \
            : InfiniopDescriptor{device_type, device_id},                                    \
              _opaque(opaque),                                                               \
              _info(info),                                                                   \
              _workspace_size(workspace_size) {}                                             \
                                                                                             \
    public:                                                                                  \
        ~Descriptor();                                                                       \
                                                                                             \
        size_t workspaceSize() const { return _workspace_size; }                             \
                                                                                             \
        static infiniStatus_t create(                                                        \
            infiniopHandle_t handle,                                                         \
            Descriptor **desc_ptr,                                                           \
            infiniopTensorDescriptor_t out_desc,                                             \
            infiniopTensorDescriptor_t start_desc,                                           \
            infiniopTensorDescriptor_t end_desc,                                             \
            infiniopTensorDescriptor_t weight_desc, /* 可为 nullptr */                     \
            float weight_scalar = 0.0f);            /* 标量模式的值 */                 \
                                                                                             \
        infiniStatus_t calculate(                                                            \
            void *workspace,                                                                 \
            size_t workspace_size,                                                           \
            void *output,                                                                    \
            const void *start,                                                               \
            const void *end,                                                                 \
            const void *weight, /* 标量模式下此指针可能为 nullptr 或被忽略 */ \
            void *stream) const;                                                             \
    };                                                                                       \
    }

#endif // __LERP_H__
