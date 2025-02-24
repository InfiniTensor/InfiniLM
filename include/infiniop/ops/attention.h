#ifndef __INFINIOP_ATTENTION_H__
#define __INFINIOP_ATTENTION_H__

#include "../operator.h"
#include "matmul.h"
#include "swiglu.h"

typedef InfiniopDescriptor *infiniopAttentionDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAttentionDescriptor(infiniopHandle_t handle,
                                                                infiniopAttentionDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t out_desc,
                                                                infiniopTensorDescriptor_t q_desc,
                                                                infiniopTensorDescriptor_t k_desc,
                                                                infiniopTensorDescriptor_t v_desc,
                                                                infiniopTensorDescriptor_t k_cache_desc,
                                                                infiniopTensorDescriptor_t v_cache_desc,
                                                                size_t pos);

__C __export infiniopStatus_t infiniopGetAttentionWorkspaceSize(infiniopAttentionDescriptor_t desc, size_t *size);

__C __export infiniopStatus_t infiniopAttention(infiniopAttentionDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *out,
                                                void const *q,
                                                void const *k,
                                                void const *v,
                                                void *k_cache,
                                                void *v_cache,
                                                void *stream);

__C __export infiniopStatus_t infiniopDestroyAttentionDescriptor(infiniopAttentionDescriptor_t desc);
#endif
