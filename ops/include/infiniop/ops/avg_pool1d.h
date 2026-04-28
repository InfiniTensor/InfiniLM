#ifndef __INFINIOP_AVG_POOL1D_API_H__
#define __INFINIOP_AVG_POOL1D_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAvgPool1dDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAvgPool1dDescriptor(
    infiniopHandle_t handle,
    infiniopAvgPool1dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    size_t kernel_size,
    size_t stride,
    size_t padding);

__INFINI_C __export infiniStatus_t infiniopGetAvgPool1dWorkspaceSize(
    infiniopAvgPool1dDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopAvgPool1d(
    infiniopAvgPool1dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAvgPool1dDescriptor(
    infiniopAvgPool1dDescriptor_t desc);

#endif
