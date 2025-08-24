#ifndef __INFINIOP_CAUSAL_SOFTMAX_API_H__
#define __INFINIOP_CAUSAL_SOFTMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopCausalSoftmaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc);

__C __export infiniStatus_t infiniopGetCausalSoftmaxWorkspaceSize(infiniopCausalSoftmaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopCausalSoftmax(
    infiniopCausalSoftmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__C __export infiniStatus_t infiniopDestroyCausalSoftmaxDescriptor(infiniopCausalSoftmaxDescriptor_t desc);

#endif
