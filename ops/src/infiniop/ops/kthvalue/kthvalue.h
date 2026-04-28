#ifndef __KTHVALUE_H__
#define __KTHVALUE_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 KthvalueInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                    \
    namespace op::kthvalue::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        KthvalueInfo _info;                                      \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            KthvalueInfo info,                                   \
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
            infiniopTensorDescriptor_t values_desc,              \
            infiniopTensorDescriptor_t indices_desc,             \
            infiniopTensorDescriptor_t input_desc,               \
            int k,                                               \
            int dim,                                             \
            int keepdim);                                        \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *values,                                        \
            void *indices,                                       \
            const void *input,                                   \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __KTHVALUE_H__
