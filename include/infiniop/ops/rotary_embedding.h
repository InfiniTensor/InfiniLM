#ifndef __INFINIOP_ROTARY_EMBEDDING_H__
#define __INFINIOP_ROTARY_EMBEDDING_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopRoPEDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t t,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table);

__C __export infiniopStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc, size_t *size);

__C __export infiniopStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *t,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream);

__C __export infiniopStatus_t infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc);

#endif
