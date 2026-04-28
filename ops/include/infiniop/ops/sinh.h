#ifndef __INFINIOP_SINH_API_H__
#define __INFINIOP_SINH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSinhDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSinhDescriptor(infiniopHandle_t handle,
                                                                infiniopSinhDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetSinhWorkspaceSize(infiniopSinhDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSinh(infiniopSinhDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySinhDescriptor(infiniopSinhDescriptor_t desc);

#endif
