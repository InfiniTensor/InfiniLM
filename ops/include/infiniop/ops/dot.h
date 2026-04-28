#ifndef __INFINIOP_DOT_API_H__
#define __INFINIOP_DOT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDotDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDotDescriptor(infiniopHandle_t handle,
                                                               infiniopDotDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t y,
                                                               infiniopTensorDescriptor_t a,
                                                               infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetDotWorkspaceSize(infiniopDotDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDot(infiniopDotDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *y,
                                               const void *a,
                                               const void *b,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDotDescriptor(infiniopDotDescriptor_t desc);

#endif
