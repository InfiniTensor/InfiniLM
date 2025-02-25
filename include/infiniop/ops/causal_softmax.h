#ifndef __INFINIOP_CAUSAL_SOFTMAX_H__
#define __INFINIOP_CAUSAL_SOFTMAX_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopCausalSoftmaxDescriptor_t;

__C __export infiniStatus_t infiniopCreateCausalSoftmaxDescriptor(infiniopHandle_t handle,
                                                                  infiniopCausalSoftmaxDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t y_desc);

__C __export infiniStatus_t infiniopGetCausalSoftmaxWorkspaceSize(infiniopCausalSoftmaxDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopCausalSoftmax(infiniopCausalSoftmaxDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *data,
                                                  void *stream);

__C __export infiniStatus_t infiniopDestroyCausalSoftmaxDescriptor(infiniopCausalSoftmaxDescriptor_t desc);

#endif
