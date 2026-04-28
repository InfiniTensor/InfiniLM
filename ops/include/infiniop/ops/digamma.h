#ifndef __INFINIOP_DIGAMMA_API_H__
#define __INFINIOP_DIGAMMA_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDigammaDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDigammaDescriptor(infiniopHandle_t handle,
                                                                   infiniopDigammaDescriptor_t *desc_ptr,
                                                                   infiniopTensorDescriptor_t y,
                                                                   infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetDigammaWorkspaceSize(infiniopDigammaDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDigamma(infiniopDigammaDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   void *y,
                                                   const void *x,
                                                   void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDigammaDescriptor(infiniopDigammaDescriptor_t desc);

#endif
