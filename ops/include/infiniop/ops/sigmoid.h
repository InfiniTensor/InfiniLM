#ifndef __INFINIOP_SIGMOID_API_H__
#define __INFINIOP_SIGMOID_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSigmoidDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSigmoidDescriptor(infiniopHandle_t handle,
                                                                   infiniopSigmoidDescriptor_t *desc_ptr,
                                                                   infiniopTensorDescriptor_t y,
                                                                   infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetSigmoidWorkspaceSize(infiniopSigmoidDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSigmoid(infiniopSigmoidDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   void *y,
                                                   const void *x,
                                                   void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySigmoidDescriptor(infiniopSigmoidDescriptor_t desc);

#endif
