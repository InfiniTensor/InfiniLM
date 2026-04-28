#ifndef __LOGCUMSUMEXP_H__
#define __LOGCUMSUMEXP_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 LogCumSumExpInfo 定义

// 宏定义：用于生成不同命名空间下的 Descriptor 类 (例如 cpu, nvidia, metax)
#define DESCRIPTOR(NAMESPACE)                                                  \
    namespace op::logcumsumexp::NAMESPACE {                                    \
    class Descriptor final : public InfiniopDescriptor {                       \
        struct Opaque;                                                         \
        Opaque *_opaque;        /* 指向后端私有实现数据 */           \
        LogCumSumExpInfo _info; /* 存储校验过的张量和算子信息 */  \
        size_t _workspace_size; /* 缓存计算所需的显存/内存大小 */ \
                                                                               \
        Descriptor(                                                            \
            Opaque *opaque,                                                    \
            LogCumSumExpInfo info,                                             \
            size_t workspace_size,                                             \
            infiniDevice_t device_type,                                        \
            int device_id)                                                     \
            : InfiniopDescriptor{device_type, device_id},                      \
              _opaque(opaque),                                                 \
              _info(info),                                                     \
              _workspace_size(workspace_size) {}                               \
                                                                               \
    public:                                                                    \
        ~Descriptor();                                                         \
                                                                               \
        size_t workspaceSize() const { return _workspace_size; }               \
                                                                               \
        static infiniStatus_t create(                                          \
            infiniopHandle_t handle,                                           \
            Descriptor **desc_ptr,                                             \
            infiniopTensorDescriptor_t y_desc,                                 \
            infiniopTensorDescriptor_t x_desc,                                 \
            int axis,                                                          \
            int exclusive,                                                     \
            int reverse);                                                      \
                                                                               \
        infiniStatus_t calculate(                                              \
            void *workspace,                                                   \
            size_t workspace_size,                                             \
            void *y,                                                           \
            const void *x,                                                     \
            void *stream) const;                                               \
    };                                                                         \
    }

#endif // __LOGCUMSUMEXP_H__
