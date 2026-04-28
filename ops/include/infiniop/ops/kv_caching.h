#ifndef __INFINIOP_KV_CACHING_API_H__
#define __INFINIOP_KV_CACHING_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopKVCachingDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateKVCachingDescriptor(
    infiniopHandle_t handle,
    infiniopKVCachingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t k_cache,
    infiniopTensorDescriptor_t v_cache,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t v,
    infiniopTensorDescriptor_t past_kv_lengths);

__INFINI_C __export infiniStatus_t infiniopGetKVCachingWorkspaceSize(infiniopKVCachingDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopKVCaching(infiniopKVCachingDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *k_cache,
                                                     void *v_cache,
                                                     const void *k,
                                                     const void *v,
                                                     const void *past_kv_lengths,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyKVCachingDescriptor(infiniopKVCachingDescriptor_t desc);

#endif
