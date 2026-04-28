#ifndef __INFINIOP_BROADCAST_TO_API_H__
#define __INFINIOP_BROADCAST_TO_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBroadcastToDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBroadcastToDescriptor(infiniopHandle_t handle,
                                                                       infiniopBroadcastToDescriptor_t *desc_ptr,
                                                                       infiniopTensorDescriptor_t y,
                                                                       infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetBroadcastToWorkspaceSize(infiniopBroadcastToDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopBroadcastTo(infiniopBroadcastToDescriptor_t desc,
                                                       void *workspace,
                                                       size_t workspace_size,
                                                       void *y,
                                                       const void *x,
                                                       void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBroadcastToDescriptor(infiniopBroadcastToDescriptor_t desc);

#endif
