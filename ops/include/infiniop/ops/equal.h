#ifndef __INFINIOP_EQUAL_API_H__
#define __INFINIOP_EQUAL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopEqualDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateEqualDescriptor(
    infiniopHandle_t handle,
    infiniopEqualDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c,
    infiniopTensorDescriptor_t a,
    infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetEqualWorkspaceSize(
    infiniopEqualDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopEqual(
    infiniopEqualDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyEqualDescriptor(
    infiniopEqualDescriptor_t desc);

#endif
