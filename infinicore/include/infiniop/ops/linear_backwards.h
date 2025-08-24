#ifndef __INFINIOP_LINEAR_BACKWARDS_API_H__
#define __INFINIOP_LINEAR_BACKWARDS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLinearBackwardsDescriptor_t;

__C __export infiniStatus_t infiniopCreateLinearBackwardsDescriptor(infiniopHandle_t handle,
                                                                   infiniopLinearBackwardsDescriptor_t *desc_ptr,
                                                                   infiniopTensorDescriptor_t grad_input_desc,
                                                                   infiniopTensorDescriptor_t grad_weight_desc,
                                                                   infiniopTensorDescriptor_t grad_bias_desc,
                                                                   infiniopTensorDescriptor_t grad_output_desc,
                                                                   infiniopTensorDescriptor_t input_desc,
                                                                   infiniopTensorDescriptor_t weight_desc);

__C __export infiniStatus_t infiniopGetLinearBackwardsWorkspaceSize(infiniopLinearBackwardsDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopLinearBackwards(infiniopLinearBackwardsDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   void *grad_input,
                                                   void *grad_weight,
                                                   void *grad_bias,
                                                   const void *grad_output,
                                                   const void *input,
                                                   const void *weight,
                                                   void *stream);

__C __export infiniStatus_t infiniopDestroyLinearBackwardsDescriptor(infiniopLinearBackwardsDescriptor_t desc);

#endif