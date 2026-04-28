#ifndef KV_CACHING_H
#define KV_CACHING_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                          \
                                                                       \
    namespace op::kv_caching::NAMESPACE {                              \
    class Descriptor final : public InfiniopDescriptor {               \
        struct Opaque;                                                 \
        Opaque *_opaque;                                               \
        KVCachingInfo _info;                                           \
        size_t _workspace_size;                                        \
                                                                       \
        Descriptor(                                                    \
            Opaque *opaque,                                            \
            KVCachingInfo info,                                        \
            size_t workspace_size,                                     \
            infiniDevice_t device_type,                                \
            int device_id)                                             \
            : InfiniopDescriptor{device_type, device_id},              \
              _opaque(opaque),                                         \
              _info(info),                                             \
              _workspace_size(workspace_size) {}                       \
                                                                       \
    public:                                                            \
        ~Descriptor();                                                 \
                                                                       \
        size_t get_workspace_size() const { return _workspace_size; }  \
                                                                       \
        static infiniStatus_t create(                                  \
            infiniopHandle_t handle,                                   \
            Descriptor **desc_ptr,                                     \
            infiniopTensorDescriptor_t k_cache,                        \
            infiniopTensorDescriptor_t v_cache,                        \
            infiniopTensorDescriptor_t k,                              \
            infiniopTensorDescriptor_t v,                              \
            infiniopTensorDescriptor_t past_kv_lengths);               \
                                                                       \
        infiniStatus_t calculate(                                      \
            void *workspace, size_t workspace_size,                    \
            void *k_cache, void *v_cache,                              \
            const void *k, const void *v, const void *past_kv_lengths, \
            void *stream) const;                                       \
    };                                                                 \
    }

#endif // KV_CACHING_H
