#ifndef PAGED_ATTENTION_PREFILL_H
#define PAGED_ATTENTION_PREFILL_H

#include "../../operator.h"
#include "info.h"

#define DESCRIPTOR(NAMESPACE)                                                   \
                                                                                \
    namespace op::paged_attention_prefill::NAMESPACE {                          \
    class Descriptor final : public InfiniopDescriptor {                        \
        struct Opaque;                                                          \
        Opaque *_opaque;                                                        \
        PagedAttentionPrefillInfo _info;                                        \
        size_t _workspace_size;                                                 \
                                                                                \
        Descriptor(                                                             \
            Opaque *opaque,                                                     \
            PagedAttentionPrefillInfo info,                                     \
            size_t workspace_size,                                              \
            infiniDevice_t device_type,                                         \
            int device_id)                                                      \
            : InfiniopDescriptor{device_type, device_id},                       \
              _opaque(opaque),                                                  \
              _info(info),                                                      \
              _workspace_size(workspace_size) {}                                \
                                                                                \
    public:                                                                     \
        ~Descriptor();                                                          \
                                                                                \
        size_t workspaceSize() const { return _workspace_size; }                \
                                                                                \
        static infiniStatus_t create(                                           \
            infiniopHandle_t handle,                                            \
            Descriptor **desc_ptr,                                              \
            infiniopTensorDescriptor_t out_desc,                                \
            infiniopTensorDescriptor_t q_desc,                                  \
            infiniopTensorDescriptor_t k_cache_desc,                            \
            infiniopTensorDescriptor_t v_cache_desc,                            \
            infiniopTensorDescriptor_t block_tables_desc,                       \
            infiniopTensorDescriptor_t seq_lens_desc,                           \
            infiniopTensorDescriptor_t cum_seq_lens_q_desc,                     \
            const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc, \
            float scale);                                                       \
                                                                                \
        infiniStatus_t calculate(                                               \
            void *workspace, size_t workspace_size,                             \
            void *out, const void *q, const void *k_cache, const void *v_cache, \
            const void *block_tables,                                           \
            const void *seq_lens,                                               \
            const void *cum_seq_lens_q,                                         \
            const void *alibi_slopes,                                           \
            void *stream) const;                                                \
    };                                                                          \
    }

#endif // PAGED_ATTENTION_PREFILL_H
