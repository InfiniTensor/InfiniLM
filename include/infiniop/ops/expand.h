#ifndef __INFINIOP_EXPAND_API_H__
#define __INFINIOP_EXPAND_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopExpandDescriptor_t;

__C __export infiniStatus_t infiniopCreateExpandDescriptor(infiniopHandle_t handle,
                                                           infiniopExpandDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t y,
                                                           infiniopTensorDescriptor_t x);

__C __export infiniStatus_t infiniopExpand(infiniopExpandDescriptor_t desc,
                                           void *y,
                                           void const *x,
                                           void *stream);

__C __export infiniStatus_t infiniopDestroyExpandDescriptor(infiniopExpandDescriptor_t desc);

#endif
