#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/paged_caching.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ALI_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#include "nvidia/paged_caching_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/paged_caching_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/paged_caching_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreatePagedCachingDescriptor(
    infiniopHandle_t handle,
    infiniopPagedCachingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {

#define CREATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        return op::paged_caching::NAMESPACE::Descriptor::create(                     \
            handle,                                                                  \
            reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor **>(desc_ptr), \
            k_cache_desc, v_cache_desc, k_desc, v_desc, slot_mapping_desc);

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
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopGetPagedCachingWorkspaceSize(
    infiniopPagedCachingDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                         \
    case CASE:                                                                                       \
        *size = reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopPagedCaching(
    infiniopPagedCachingDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *k_cache, void *v_cache,
    const void *k, const void *v,
    const void *slot_mapping,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                                \
        return reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, k_cache, v_cache, k, v, slot_mapping, stream);

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
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__INFINI_C infiniStatus_t infiniopDestroyPagedCachingDescriptor(
    infiniopPagedCachingDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                   \
    case CASE:                                                                     \
        delete reinterpret_cast<op::paged_caching::NAMESPACE::Descriptor *>(desc); \
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
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
