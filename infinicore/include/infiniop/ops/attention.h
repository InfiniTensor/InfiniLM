#ifndef __INFINIOP_ATTENTION_API_H__
#define __INFINIOP_ATTENTION_API_H__

#include "../operator_descriptor.h"
#include "gemm.h"
#include "swiglu.h"

typedef struct InfiniopDescriptor *infiniopAttentionDescriptor_t;

__C __export infiniStatus_t infiniopCreateAttentionDescriptor(infiniopHandle_t handle,
                                                              infiniopAttentionDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t out_desc,
                                                              infiniopTensorDescriptor_t q_desc,
                                                              infiniopTensorDescriptor_t k_desc,
                                                              infiniopTensorDescriptor_t v_desc,
                                                              infiniopTensorDescriptor_t k_cache_desc,
                                                              infiniopTensorDescriptor_t v_cache_desc,
                                                              size_t pos);

__C __export infiniStatus_t infiniopGetAttentionWorkspaceSize(infiniopAttentionDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAttention(infiniopAttentionDescriptor_t desc,
                                              void *workspace,
                                              size_t workspace_size,
                                              void *out,
                                              const void *q,
                                              const void *k,
                                              const void *v,
                                              void *k_cache,
                                              void *v_cache,
                                              void *stream);

__C __export infiniStatus_t infiniopDestroyAttentionDescriptor(infiniopAttentionDescriptor_t desc);
#endif
