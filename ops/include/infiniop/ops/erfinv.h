#ifndef __INFINIOP_ERFINV_API_H__
#define __INFINIOP_ERFINV_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopErfinvDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateErfinvDescriptor(infiniopHandle_t handle,
                                                                  infiniopErfinvDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t y,
                                                                  infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetErfinvWorkspaceSize(infiniopErfinvDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopErfinv(infiniopErfinvDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *y,
                                                  const void *x,
                                                  void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyErfinvDescriptor(infiniopErfinvDescriptor_t desc);

#endif
