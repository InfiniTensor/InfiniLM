#ifndef __INDEX_COPY_H__
#define __INDEX_COPY_H__

#include "../../operator.h"
#include "info.h" // 引用 IndexCopyInfo 定义
#define DESCRIPTOR(NAMESPACE)                                                       \
                                                                                    \
    namespace op::index_copy::NAMESPACE {                                           \
    class Descriptor final : public InfiniopDescriptor {                            \
        struct Opaque;                                                              \
        Opaque *_opaque;                                                            \
        IndexCopyInfo _info;                                                        \
        size_t _workspace_size;                                                     \
                                                                                    \
        Descriptor(                                                                 \
            Opaque *opaque,                                                         \
            IndexCopyInfo info,                                                     \
            size_t workspace_size,                                                  \
            infiniDevice_t device_type,                                             \
            int device_id)                                                          \
            : InfiniopDescriptor{device_type, device_id},                           \
              _opaque(opaque),                                                      \
              _info(info),                                                          \
              _workspace_size(workspace_size) {}                                    \
                                                                                    \
    public:                                                                         \
        ~Descriptor();                                                              \
                                                                                    \
        size_t workspaceSize() const { return _workspace_size; }                    \
                                                                                    \
        static infiniStatus_t create(                                               \
            infiniopHandle_t handle,                                                \
            Descriptor **desc_ptr,                                                  \
            infiniopTensorDescriptor_t out_desc,                                    \
            infiniopTensorDescriptor_t in_desc,                                     \
            int64_t dim,                                                            \
            infiniopTensorDescriptor_t index_desc,                                  \
            infiniopTensorDescriptor_t source_desc); /* 注意：移除了 alpha */ \
                                                                                    \
        infiniStatus_t calculate(                                                   \
            void *workspace,                                                        \
            size_t workspace_size,                                                  \
            void *output,                                                           \
            const void *input,                                                      \
            const void *index,                                                      \
            const void *source,                                                     \
            void *stream) const;                                                    \
    };                                                                              \
    }

#endif // __INDEX_COPY_H__
