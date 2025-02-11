#ifndef __INFINIOP_EXPAND_H__
#define __INFINIOP_EXPAND_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopExpandDescriptor_t;

__C __export infiniopStatus_t infiniopCreateExpandDescriptor(infiniopHandle_t handle,
                                                             infiniopExpandDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t y,
                                                             infiniopTensorDescriptor_t x);

__C __export infiniopStatus_t infiniopExpand(infiniopExpandDescriptor_t desc,
                                             void *y,
                                             void const *x,
                                             void *stream);

__C __export infiniopStatus_t infiniopDestroyExpandDescriptor(infiniopExpandDescriptor_t desc);

#endif
