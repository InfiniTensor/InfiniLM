#ifndef __INFINIOP_RELU_H__
#define __INFINIOP_RELU_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopReluDescriptor_t;

__C __export infiniStatus_t infiniopCreateReluDescriptor(infiniopHandle_t handle,
                                                         infiniopReluDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y,
                                                         infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopRelu(infiniopReluDescriptor_t desc,
                                         void *y,
                                         void const *x,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyReluDescriptor(infiniopReluDescriptor_t desc);

#endif
