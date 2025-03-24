#ifndef __INFINIOP_GLOBAL_AVG_POOL_API_H__
#define __INFINIOP_GLOBAL_AVG_POOL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGlobalAvgPoolDescriptor_t;

__C __export infiniStatus_t infiniopCreateGlobalAvgPoolDescriptor(infiniopHandle_t handle,
                                                                  infiniopGlobalAvgPoolDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t y,
                                                                  infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopGetGlobalAvgPoolWorkspaceSize(infiniopGlobalAvgPoolDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGlobalAvgPool(infiniopGlobalAvgPoolDescriptor_t desc,
                                                  void *workspace, size_t workspace_size,
                                                  void *y, void const *x, void *stream);

__C __export infiniStatus_t infiniopDestroyGlobalAvgPoolDescriptor(infiniopGlobalAvgPoolDescriptor_t desc);

#endif
