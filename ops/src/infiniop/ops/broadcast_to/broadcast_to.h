#ifndef __BROADCAST_TO_H__
#define __BROADCAST_TO_H__

#include "../../operator.h"
#include "info.h" // 引用对应的 BroadcastToInfo 定义
#include <vector>

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                            \
    namespace op::broadcast_to::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {                 \
        struct Opaque;                                                   \
        Opaque *_opaque;                                                 \
        BroadcastToInfo _info;                                           \
        size_t _workspace_size;                                          \
                                                                         \
        Descriptor(                                                      \
            Opaque *opaque,                                              \
            BroadcastToInfo info,                                        \
            size_t workspace_size,                                       \
            infiniDevice_t device_type,                                  \
            int device_id)                                               \
            : InfiniopDescriptor{device_type, device_id},                \
              _opaque(opaque),                                           \
              _info(info),                                               \
              _workspace_size(workspace_size) {}                         \
                                                                         \
    public:                                                              \
        ~Descriptor();                                                   \
                                                                         \
        size_t workspaceSize() const { return _workspace_size; }         \
                                                                         \
        static infiniStatus_t create(                                    \
            infiniopHandle_t handle,                                     \
            Descriptor **desc_ptr,                                       \
            infiniopTensorDescriptor_t out_desc,                         \
            const std::vector<infiniopTensorDescriptor_t> &input_descs); \
                                                                         \
        infiniStatus_t calculate(                                        \
            void *workspace,                                             \
            size_t workspace_size,                                       \
            void *output,                                                \
            const std::vector<const void *> &inputs,                     \
            void *stream) const;                                         \
    };                                                                   \
    }

#endif // __BROADCAST_TO_H__
