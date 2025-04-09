#ifndef __INFINIOP_ROPE_API_H__
#define __INFINIOP_ROPE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRoPEDescriptor_t;

__C __export infiniStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table);

__C __export infiniStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream);

__C __export infiniStatus_t infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc);

#endif
