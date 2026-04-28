#ifndef __INFINIOP_Atanh_API_H__
#define __INFINIOP_Atanh_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAtanhDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAtanhDescriptor(infiniopHandle_t handle,
                                                                 infiniopAtanhDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t a);

__INFINI_C __export infiniStatus_t infiniopGetAtanhWorkspaceSize(infiniopAtanhDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAtanh(infiniopAtanhDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *a,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAtanhDescriptor(infiniopAtanhDescriptor_t desc);

#endif
