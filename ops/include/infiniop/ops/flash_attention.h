#ifndef __INFINIOP_FLASH_ATTENTION_API_H__
#define __INFINIOP_FLASH_ATTENTION_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFlashAttentionDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFlashAttentionDescriptor(
    infiniopHandle_t handle,
    infiniopFlashAttentionDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t total_kv_len,
    float scale,
    char is_causal);

__INFINI_C __export infiniStatus_t infiniopGetFlashAttentionWorkspaceSize(
    infiniopFlashAttentionDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopFlashAttention(
    infiniopFlashAttentionDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *q,
    const void *k,
    const void *v,
    const void *total_kv_len,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFlashAttentionDescriptor(
    infiniopFlashAttentionDescriptor_t desc);
#endif
