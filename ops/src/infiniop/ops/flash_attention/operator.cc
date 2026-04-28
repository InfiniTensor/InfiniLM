#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/flash_attention.h"

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
#include "ninetoothed/descriptor.h"
#endif
#endif

__INFINI_C infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t total_kv_len,
    float scale,
    char is_causal) {

#define CREATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        return op::flash_attention::NAMESPACE::Descriptor::create(                     \
            handle,                                                                    \
            reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                                  \
            q_desc,                                                                    \
            k_desc,                                                                    \
            v_desc,                                                                    \
            total_kv_len,                                                              \
            scale,                                                                     \
            is_causal);

    switch (handle->device) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        CREATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc,
    size_t *size) {

#define GET_SIZE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                             \
        *size = reinterpret_cast<const op::flash_attention::NAMESPACE::Descriptor *>(desc) \
                    ->get_workspace_size();                                                \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        GET_SIZE(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET_SIZE
}

__INFINI_C infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                            \
        return reinterpret_cast<const op::flash_attention::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, q, k, v, total_kv_len, stream);

    switch (desc->device_type) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        CALCULATE(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyFlashAttentionDescriptor(
    infiniopFlashAttentionDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                     \
    case CASE:                                                                       \
        delete reinterpret_cast<op::flash_attention::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#if defined(ENABLE_NINETOOTHED)
#if defined(ENABLE_NVIDIA_API)
        DESTROY(INFINI_DEVICE_NVIDIA, ninetoothed);
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
