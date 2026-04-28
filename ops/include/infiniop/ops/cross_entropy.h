#ifndef __INFINIOP_CROSS_ENTROPY_API_H__
#define __INFINIOP_CROSS_ENTROPY_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCrossEntropyDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateCrossEntropyDescriptor(
    infiniopHandle_t handle,
    infiniopCrossEntropyDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t target_desc);

__INFINI_C __export infiniStatus_t infiniopGetCrossEntropyWorkspaceSize(
    infiniopCrossEntropyDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopCrossEntropy(
    infiniopCrossEntropyDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *target,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyCrossEntropyDescriptor(
    infiniopCrossEntropyDescriptor_t desc);

#endif
