#ifndef __INFINIOP_REARRANGE_H__
#define __INFINIOP_REARRANGE_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopRearrangeDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRearrangeDescriptor(infiniopHandle_t handle,
                                                                infiniopRearrangeDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t dst,
                                                                infiniopTensorDescriptor_t src);

__C __export infiniopStatus_t infiniopRearrange(infiniopRearrangeDescriptor_t desc, void *dst, void const *src, void *stream);

__C __export infiniopStatus_t infiniopDestroyRearrangeDescriptor(infiniopRearrangeDescriptor_t desc);
#endif
