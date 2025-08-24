#ifndef __INFINIOP_REARRANGE_API_H__
#define __INFINIOP_REARRANGE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRearrangeDescriptor_t;

__C __export infiniStatus_t infiniopCreateRearrangeDescriptor(
    infiniopHandle_t handle,
    infiniopRearrangeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src);

__C __export infiniStatus_t infiniopRearrange(
    infiniopRearrangeDescriptor_t desc,
    void *dst,
    const void *src,
    void *stream);

__C __export infiniStatus_t infiniopDestroyRearrangeDescriptor(
    infiniopRearrangeDescriptor_t desc);

#endif
