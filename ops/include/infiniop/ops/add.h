#ifndef __INFINIOP_ADD_API_H__
#define __INFINIOP_ADD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAddDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAddDescriptor(infiniopHandle_t handle,
                                                               infiniopAddDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t c,
                                                               infiniopTensorDescriptor_t a,
                                                               infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetAddWorkspaceSize(infiniopAddDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAdd(infiniopAddDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *c,
                                               const void *a,
                                               const void *b,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAddDescriptor(infiniopAddDescriptor_t desc);

#endif
