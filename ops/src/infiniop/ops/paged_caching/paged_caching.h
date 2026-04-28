#ifndef PAGED_CACHING_H
#define PAGED_CACHING_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                    \
                                                                 \
    namespace op::paged_caching::NAMESPACE {                     \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        PagedCachingInfo _info;                                  \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            Opaque *opaque,                                      \
            PagedCachingInfo info,                               \
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
            infiniopTensorDescriptor_t k_cache_desc,             \
            infiniopTensorDescriptor_t v_cache_desc,             \
            infiniopTensorDescriptor_t k_desc,                   \
            infiniopTensorDescriptor_t v_desc,                   \
            infiniopTensorDescriptor_t slot_mapping_desc);       \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *k_cache, void *v_cache,                        \
            const void *k, const void *v,                        \
            const void *slot_mapping,                            \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // PAGED_CACHING_H
