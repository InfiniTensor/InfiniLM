#ifndef __INFINIOP_AVG_POOL_H__
#define __INFINIOP_AVG_POOL_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopAvgPoolDescriptor_t;

__C __export infiniStatus_t infiniopCreateAvgPoolDescriptor(infiniopHandle_t handle,
                                                            infiniopAvgPoolDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y,
                                                            infiniopTensorDescriptor_t x,
                                                            size_t const *kernel_shape,
                                                            size_t const *pads,
                                                            ptrdiff_t const *strides,
                                                            size_t n);

__C __export infiniStatus_t infiniopGetAvgPoolWorkspaceSize(infiniopAvgPoolDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopAvgPool(infiniopAvgPoolDescriptor_t desc,
                                            void *workspace, size_t workspace_size,
                                            void *y, void const *x, void *stream);

__C __export infiniStatus_t infiniopDestroyAvgPoolDescriptor(infiniopAvgPoolDescriptor_t desc);
#endif
