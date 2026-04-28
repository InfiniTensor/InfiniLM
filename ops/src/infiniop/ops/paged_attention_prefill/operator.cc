#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/paged_attention_prefill.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_HYGON_API)
#include "nvidia/paged_attention_prefill_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/paged_attention_prefill_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/paged_attention_prefill_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreatePagedAttentionPrefillDescriptor(
    infiniopHandle_t handle,
    infiniopPagedAttentionPrefillDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t seq_lens_desc,
    infiniopTensorDescriptor_t cum_seq_lens_q_desc,
    infiniopTensorDescriptor_t alibi_slopes_desc,
    float scale) {

    infiniopTensorDescriptor_t alibi_opt = (alibi_slopes_desc == nullptr) ? nullptr : alibi_slopes_desc;

#define CREATE(CASE, NAMESPACE)                                                                \
    case CASE:                                                                                 \
        return op::paged_attention_prefill::NAMESPACE::Descriptor::create(                     \
            handle,                                                                            \
            reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, q_desc, k_cache_desc, v_cache_desc, block_tables_desc,                   \
            seq_lens_desc, cum_seq_lens_q_desc, alibi_opt, scale);

    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopGetPagedAttentionPrefillWorkspaceSize(
    infiniopPagedAttentionPrefillDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                                   \
    case CASE:                                                                                                 \
        *size = reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopPagedAttentionPrefill(
    infiniopPagedAttentionPrefillDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables,
    const void *seq_lens,
    const void *cum_seq_lens_q,
    const void *alibi_slopes,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                          \
        return reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, q, k_cache, v_cache, block_tables,                          \
            seq_lens, cum_seq_lens_q, alibi_slopes, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopDestroyPagedAttentionPrefillDescriptor(
    infiniopPagedAttentionPrefillDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                             \
    case CASE:                                                                               \
        delete reinterpret_cast<op::paged_attention_prefill::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia)
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia)
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
