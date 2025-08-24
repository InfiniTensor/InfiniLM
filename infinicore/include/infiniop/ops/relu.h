#ifndef __INFINIOP_RELU_API_H__
#define __INFINIOP_RELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopReluDescriptor_t;

__C __export infiniStatus_t infiniopCreateReluDescriptor(infiniopHandle_t handle,
                                                         infiniopReluDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopRelu(infiniopReluDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         const void *x,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyReluDescriptor(infiniopReluDescriptor_t desc);

#endif
