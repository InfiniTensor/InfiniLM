#ifndef __INFINIOP_SUB_API_H__
#define __INFINIOP_SUB_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSubDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSubDescriptor(infiniopHandle_t handle,
                                                               infiniopSubDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t c,
                                                               infiniopTensorDescriptor_t a,
                                                               infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetSubWorkspaceSize(infiniopSubDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSub(infiniopSubDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *c,
                                               const void *a,
                                               const void *b,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySubDescriptor(infiniopSubDescriptor_t desc);

#endif
