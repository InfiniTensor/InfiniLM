#ifndef __INFINIOP_INNER_API_H__
#define __INFINIOP_INNER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopInnerDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateInnerDescriptor(
    infiniopHandle_t handle,
    infiniopInnerDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t other_desc);

__INFINI_C __export infiniStatus_t infiniopGetInnerWorkspaceSize(infiniopInnerDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopInner(
    infiniopInnerDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *input,
    const void *other,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyInnerDescriptor(infiniopInnerDescriptor_t desc);

#endif
