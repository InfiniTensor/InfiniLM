#ifndef __INFINIOP_OPS_SOFTPLUS_API_H__
#define __INFINIOP_OPS_SOFTPLUS_API_H__
#include "../tensor_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSoftplusDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateSoftplusDescriptor(
    infiniopHandle_t handle,
    infiniopSoftplusDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    float beta,
    float threshold);

__INFINI_C __export infiniStatus_t infiniopGetSoftplusWorkspaceSize(
    infiniopSoftplusDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopSoftplus(
    infiniopSoftplusDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySoftplusDescriptor(
    infiniopSoftplusDescriptor_t desc);

#endif // __INFINIOP_OPS_SOFTPLUS_API_H__
