#ifndef __INFINIOP_LINEAR_API_H__
#define __INFINIOP_LINEAR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLinearDescriptor_t;

__C __export infiniStatus_t infiniopCreateLinearDescriptor(infiniopHandle_t handle,
                                                          infiniopLinearDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t output_desc,
                                                          infiniopTensorDescriptor_t input_desc,
                                                          infiniopTensorDescriptor_t weight_desc,
                                                          infiniopTensorDescriptor_t bias_desc);

__C __export infiniStatus_t infiniopGetLinearWorkspaceSize(infiniopLinearDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLinear(infiniopLinearDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *output,
                                          const void *input,
                                          const void *weight,
                                          const void *bias,
                                          void *stream);

__C __export infiniStatus_t infiniopDestroyLinearDescriptor(infiniopLinearDescriptor_t desc);

#endif