#ifndef __INFINIOP_ASINH_API_H__
#define __INFINIOP_ASINH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAsinhDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAsinhDescriptor(infiniopHandle_t handle,
                                                                 infiniopAsinhDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetAsinhWorkspaceSize(infiniopAsinhDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAsinh(infiniopAsinhDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *x,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAsinhDescriptor(infiniopAsinhDescriptor_t desc);

#endif
