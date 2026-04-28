#ifndef __INFINIOP_ROPE_API_H__
#define __INFINIOP_ROPE_API_H__

#include "../operator_descriptor.h"

typedef enum {
    INFINIOP_ROPE_ALGO_GPT_J = 0,    // GPT-J style RoPE algorithm (Interleave even and odd dimensions)
    INFINIOP_ROPE_ALGO_GPT_NEOX = 1, // GPT-NeoX style RoPE algorithm (First half dimensions for sin, second half for cos)
    // Count
    INFINIOP_ROPE_ALGO_COUNT = 2,
} infiniopRoPEAlgo_t;

typedef struct InfiniopDescriptor *infiniopRoPEDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table,
    infiniopRoPEAlgo_t algo);

__INFINI_C __export infiniStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void const *pos_ids,
    void const *sin_table,
    void const *cos_table,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc);

#endif
