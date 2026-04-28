#ifndef __INFINIOP_HARDSWISH_API_H__
#define __INFINIOP_HARDSWISH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopHardSwishDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHardSwishDescriptor(
    infiniopHandle_t handle,
    infiniopHardSwishDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input);

__INFINI_C __export infiniStatus_t infiniopGetHardSwishWorkspaceSize(
    infiniopHardSwishDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopHardSwish(
    infiniopHardSwishDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHardSwishDescriptor(
    infiniopHardSwishDescriptor_t desc);

#endif
