#ifndef __INFINIOP_LOGADDEXP_API_H__
#define __INFINIOP_LOGADDEXP_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogAddExpDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLogAddExpDescriptor(infiniopHandle_t handle,
                                                                     infiniopLogAddExpDescriptor_t *desc_ptr,
                                                                     infiniopTensorDescriptor_t c,
                                                                     infiniopTensorDescriptor_t a,
                                                                     infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetLogAddExpWorkspaceSize(infiniopLogAddExpDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLogAddExp(infiniopLogAddExpDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *c,
                                                     const void *a,
                                                     const void *b,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLogAddExpDescriptor(infiniopLogAddExpDescriptor_t desc);

#endif // __INFINIOP_LOGADDEXP_API_H__
