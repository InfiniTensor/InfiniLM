#ifndef __INFINIOP_TAKE_API_H__
#define __INFINIOP_TAKE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTakeDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTakeDescriptor(infiniopHandle_t handle,
                                                                infiniopTakeDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t output,
                                                                infiniopTensorDescriptor_t input,
                                                                infiniopTensorDescriptor_t indices);

__INFINI_C __export infiniStatus_t infiniopGetTakeWorkspaceSize(infiniopTakeDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTake(infiniopTakeDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *output,
                                                const void *input,
                                                const void *indices,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTakeDescriptor(infiniopTakeDescriptor_t desc);

#endif
