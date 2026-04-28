#ifndef __INFINIOP_AVG_POOL3D_API_H__
#define __INFINIOP_AVG_POOL3D_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAvgPool3dDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAvgPool3dDescriptor(infiniopHandle_t handle,
                                                                     infiniopAvgPool3dDescriptor_t *desc_ptr,
                                                                     infiniopTensorDescriptor_t y,
                                                                     infiniopTensorDescriptor_t x,
                                                                     void *kernel_size,
                                                                     void *stride,
                                                                     void *padding);

__INFINI_C __export infiniStatus_t infiniopGetAvgPool3dWorkspaceSize(infiniopAvgPool3dDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAvgPool3d(infiniopAvgPool3dDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *y,
                                                     const void *x,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAvgPool3dDescriptor(infiniopAvgPool3dDescriptor_t desc);

#endif
