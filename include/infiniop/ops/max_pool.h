#ifndef __INFINIOP_MAX_POOL_H__
#define __INFINIOP_MAX_POOL_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopMaxPoolDescriptor_t;

__C __export infiniStatus_t infiniopCreateMaxPoolDescriptor(infiniopHandle_t handle,
                                                            infiniopMaxPoolDescriptor_t *desc_ptr,
                                                            infiniopTensorDescriptor_t y,
                                                            infiniopTensorDescriptor_t x,
                                                            size_t const *kernel_shape,
                                                            size_t const *pads,
                                                            ptrdiff_t const *strides,
                                                            size_t n);

__C __export infiniStatus_t infiniopGetMaxPoolWorkspaceSize(infiniopMaxPoolDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMaxPool(infiniopMaxPoolDescriptor_t desc,
                                            void *workspace, size_t workspace_size,
                                            void *y, void const *x, void *stream);

__C __export infiniStatus_t infiniopDestroyMaxPoolDescriptor(infiniopMaxPoolDescriptor_t desc);
#endif
