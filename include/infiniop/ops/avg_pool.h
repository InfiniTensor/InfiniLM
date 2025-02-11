#ifndef __INFINIOP_AVG_POOL_H__
#define __INFINIOP_AVG_POOL_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopAvgPoolDescriptor_t;

__C __export infiniopStatus_t infiniopCreateAvgPoolDescriptor(infiniopHandle_t handle,
                                                              infiniopAvgPoolDescriptor_t *desc_ptr,
                                                              infiniopTensorDescriptor_t y,
                                                              infiniopTensorDescriptor_t x,
                                                              uint64_t const *kernel_shape,
                                                              uint64_t const *pads,
                                                              int64_t const *strides,
                                                              uint64_t n);

__C __export infiniopStatus_t infiniopGetAvgPoolWorkspaceSize(infiniopAvgPoolDescriptor_t desc, size_t *size);

__C __export infiniopStatus_t infiniopAvgPool(infiniopAvgPoolDescriptor_t desc,
                                              void *workspace, size_t workspace_size,
                                              void *y, void const *x, void *stream);

__C __export infiniopStatus_t infiniopDestroyAvgPoolDescriptor(infiniopAvgPoolDescriptor_t desc);
#endif
