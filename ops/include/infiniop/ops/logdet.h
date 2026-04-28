#ifndef __INFINIOP_LOGDET_API_H__
#define __INFINIOP_LOGDET_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogdetDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLogdetDescriptor(infiniopHandle_t handle,
                                                                  infiniopLogdetDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t y,
                                                                  infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetLogdetWorkspaceSize(infiniopLogdetDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLogdet(infiniopLogdetDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *y,
                                                  const void *x,
                                                  void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLogdetDescriptor(infiniopLogdetDescriptor_t desc);

#endif
