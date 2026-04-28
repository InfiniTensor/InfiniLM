#ifndef __FLOAT_POWER_H__
#define __FLOAT_POWER_H__

#include "../../operator.h"
#include "info.h"

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                              \
    namespace op::float_power::NAMESPACE {                                 \
    class Descriptor final : public InfiniopDescriptor {                   \
        struct Opaque;                                                     \
        Opaque *_opaque;                                                   \
        FloatPowerInfo _info;                                              \
        size_t _workspace_size;                                            \
                                                                           \
        Descriptor(                                                        \
            Opaque *opaque,                                                \
            FloatPowerInfo info,                                           \
            size_t workspace_size,                                         \
            infiniDevice_t device_type,                                    \
            int device_id)                                                 \
            : InfiniopDescriptor{device_type, device_id},                  \
              _opaque(opaque),                                             \
              _info(info),                                                 \
              _workspace_size(workspace_size) {}                           \
                                                                           \
    public:                                                                \
        ~Descriptor();                                                     \
                                                                           \
        size_t workspaceSize() const { return _workspace_size; }           \
                                                                           \
        /* [修改] 增加 exponent 张量描述符 和 scalar_exponent */ \
        static infiniStatus_t create(                                      \
            infiniopHandle_t handle,                                       \
            Descriptor **desc_ptr,                                         \
            infiniopTensorDescriptor_t y,                                  \
            infiniopTensorDescriptor_t x,                                  \
            infiniopTensorDescriptor_t exponent,                           \
            float scalar_exponent);                                        \
                                                                           \
        /* [修改] 增加 exponent 数据指针 */                        \
        infiniStatus_t calculate(                                          \
            void *workspace,                                               \
            size_t workspace_size,                                         \
            void *y,                                                       \
            const void *x,                                                 \
            const void *exponent,                                          \
            void *stream) const;                                           \
    };                                                                     \
    }

#endif // __FLOAT_POWER_H__
