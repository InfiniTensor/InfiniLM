#ifndef __INFINIOP_SELU_API_H__
#define __INFINIOP_SELU_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSeluDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSeluDescriptor(infiniopHandle_t handle,
                                                                infiniopSeluDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetSeluWorkspaceSize(infiniopSeluDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSelu(infiniopSeluDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySeluDescriptor(infiniopSeluDescriptor_t desc);

#endif
