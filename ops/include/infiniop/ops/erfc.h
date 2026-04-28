#ifndef __INFINIOP_ERFC_API_H__
#define __INFINIOP_ERFC_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopErfcDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateErfcDescriptor(infiniopHandle_t handle,
                                                                infiniopErfcDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetErfcWorkspaceSize(infiniopErfcDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopErfc(infiniopErfcDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyErfcDescriptor(infiniopErfcDescriptor_t desc);

#endif
