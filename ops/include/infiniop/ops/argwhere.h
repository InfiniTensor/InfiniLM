#ifndef __INFINIOP_ARGWHERE_API_H__
#define __INFINIOP_ARGWHERE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopArgwhereDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateArgwhereDescriptor(
    infiniopHandle_t handle,
    infiniopArgwhereDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t input_desc);

__INFINI_C __export infiniStatus_t infiniopGetArgwhereWorkspaceSize(
    infiniopArgwhereDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopArgwhere(
    infiniopArgwhereDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void **output,
    size_t *count,
    const void *input,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyArgwhereDescriptor(
    infiniopArgwhereDescriptor_t desc);

#endif // __INFINIOP_ARGWHERE_API_H__
