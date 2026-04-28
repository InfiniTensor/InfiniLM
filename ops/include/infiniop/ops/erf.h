#ifndef __INFINIOP_ERF_API_H__
#define __INFINIOP_ERF_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopErfDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateErfDescriptor(infiniopHandle_t handle,
                                                               infiniopErfDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t y,
                                                               infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetErfWorkspaceSize(infiniopErfDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopErf(infiniopErfDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *y,
                                               const void *x,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyErfDescriptor(infiniopErfDescriptor_t desc);

#endif
