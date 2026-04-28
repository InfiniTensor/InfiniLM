#ifndef AFFINE_GRID_H
#define AFFINE_GRID_H

#include "../../operator.h"
#include "info.h"

// 宏定义：用于生成不同命名空间下的 Descriptor 类
#define DESCRIPTOR(NAMESPACE)                                      \
                                                                   \
    namespace op::affine_grid::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {           \
        struct Opaque;                                             \
        Opaque *_opaque;                                           \
        AffineGridInfo _info;                                      \
        size_t _workspace_size;                                    \
                                                                   \
        Descriptor(                                                \
            Opaque *opaque,                                        \
            AffineGridInfo info,                                   \
            size_t workspace_size,                                 \
            infiniDevice_t device_type,                            \
            int device_id)                                         \
            : InfiniopDescriptor{device_type, device_id},          \
              _opaque(opaque),                                     \
              _info(info),                                         \
              _workspace_size(workspace_size) {}                   \
                                                                   \
    public:                                                        \
        ~Descriptor();                                             \
                                                                   \
        size_t workspaceSize() const { return _workspace_size; }   \
                                                                   \
        static infiniStatus_t create(                              \
            infiniopHandle_t handle,                               \
            Descriptor **desc_ptr,                                 \
            infiniopTensorDescriptor_t out_desc,                   \
            infiniopTensorDescriptor_t in_desc,                    \
            bool align_corners); /* 增加 align_corners 参数 */ \
                                                                   \
        infiniStatus_t calculate(                                  \
            void *workspace,                                       \
            size_t workspace_size,                                 \
            void *output,                                          \
            const void *input,                                     \
            void *stream) const;                                   \
    };                                                             \
    }

#endif // AFFINE_GRID_H
