#ifndef __INFINIOP_TANH_API_H__
#define __INFINIOP_TANH_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTanhDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTanhDescriptor(infiniopHandle_t handle,
                                                                infiniopTanhDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t output,
                                                                infiniopTensorDescriptor_t input);

__INFINI_C __export infiniStatus_t infiniopGetTanhWorkspaceSize(infiniopTanhDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTanh(infiniopTanhDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *output,
                                                const void *input,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTanhDescriptor(infiniopTanhDescriptor_t desc);

#endif
