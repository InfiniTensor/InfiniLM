#ifndef __INFINIOP_PRELU_API_H__
#define __INFINIOP_PRELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopPreluDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePreluDescriptor(infiniopHandle_t handle,
                                                                 infiniopPreluDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x,
                                                                 infiniopTensorDescriptor_t weight);

__INFINI_C __export infiniStatus_t infiniopGetPreluWorkspaceSize(infiniopPreluDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPrelu(infiniopPreluDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *x,
                                                 const void *weight,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPreluDescriptor(infiniopPreluDescriptor_t desc);

#endif
