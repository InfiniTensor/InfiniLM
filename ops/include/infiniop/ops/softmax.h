#ifndef __INFINIOP_SOFTMAX_API_H__
#define __INFINIOP_SOFTMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSoftmaxDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    int axis);

__INFINI_C __export infiniStatus_t infiniopGetSoftmaxWorkspaceSize(infiniopSoftmaxDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSoftmax(
    infiniopSoftmaxDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySoftmaxDescriptor(infiniopSoftmaxDescriptor_t desc);

#endif
