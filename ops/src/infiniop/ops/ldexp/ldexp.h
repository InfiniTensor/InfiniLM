#ifndef __LDEXP_H__
#define __LDEXP_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 LdexpInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                                     \
    namespace op::ldexp::NAMESPACE {                                              \
    class Descriptor final : public InfiniopDescriptor {                          \
        struct Opaque;                                                            \
        Opaque *_opaque;                                                          \
        LdexpInfo _info;                                                          \
        size_t _workspace_size;                                                   \
                                                                                  \
        Descriptor(                                                               \
            Opaque *opaque,                                                       \
            LdexpInfo info,                                                       \
            size_t workspace_size,                                                \
            infiniDevice_t device_type,                                           \
            int device_id)                                                        \
            : InfiniopDescriptor{device_type, device_id},                         \
              _opaque(opaque),                                                    \
              _info(info),                                                        \
              _workspace_size(workspace_size) {}                                  \
                                                                                  \
    public:                                                                       \
        ~Descriptor();                                                            \
                                                                                  \
        size_t workspaceSize() const { return _workspace_size; }                  \
                                                                                  \
        static infiniStatus_t create(                                             \
            infiniopHandle_t handle,                                              \
            Descriptor **desc_ptr,                                                \
            infiniopTensorDescriptor_t y_desc,                                    \
            infiniopTensorDescriptor_t x_desc,                                    \
            infiniopTensorDescriptor_t exp_desc);                                 \
                                                                                  \
        infiniStatus_t calculate(                                                 \
            void *workspace,                                                      \
            size_t workspace_size,                                                \
            void *output,                                                         \
            const void *x,                                                        \
            const void *exp,                                                      \
            void *stream) const;                                                  \
        /* 为了兼容 Element-wise 框架通常传入 vector 的接口形式 */ \
        infiniStatus_t calculate(                                                 \
            void *workspace,                                                      \
            size_t workspace_size,                                                \
            void *output,                                                         \
            std::vector<const void *> inputs,                                     \
            void *stream) const;                                                  \
    };                                                                            \
    }

#endif // __LDEXP_H__
