#ifndef __INFINIOP_LAYER_NORM_API_H__
#define __INFINIOP_LAYER_NORM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLayerNormDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLayerNormDescriptor(
    infiniopHandle_t handle,
    infiniopLayerNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_standardization_desc,
    infiniopTensorDescriptor_t input_std_deviation_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t bias_desc,
    float eps);

__INFINI_C __export infiniStatus_t infiniopGetLayerNormWorkspaceSize(infiniopLayerNormDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLayerNorm(infiniopLayerNormDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *output,
                                                     void *input_standardization,
                                                     void *input_std_deviation,
                                                     const void *input,
                                                     const void *weight,
                                                     const void *bias,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLayerNormDescriptor(infiniopLayerNormDescriptor_t desc);

#endif
