#ifndef __INFINIOP_KRON_API_H__
#define __INFINIOP_KRON_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopKronDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateKronDescriptor(infiniopHandle_t handle,
                                                                infiniopKronDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x1,
                                                                infiniopTensorDescriptor_t x2);

__INFINI_C __export infiniStatus_t infiniopGetKronWorkspaceSize(infiniopKronDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopKron(infiniopKronDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x1,
                                                const void *x2,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyKronDescriptor(infiniopKronDescriptor_t desc);

#endif
