#ifndef __INDEX_ADD_H__
#define __INDEX_ADD_H__

#include "../../operator.h"
#include "info.h" // 引用 IndexAddInfo 定义 (需自行定义，包含 dim, alpha 等)
#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::index_add::NAMESPACE {                         \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        IndexAddInfo _info;                                      \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            IndexAddInfo info,                                   \
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
            infiniopTensorDescriptor_t in_desc,                  \
            int64_t dim,                                         \
            infiniopTensorDescriptor_t index_desc,               \
            infiniopTensorDescriptor_t source_desc,              \
            float alpha);                                        \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace,                                     \
            size_t workspace_size,                               \
            void *output,                                        \
            const void *input,                                   \
            const void *index,                                   \
            const void *source,                                  \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __INDEX_ADD_H__
