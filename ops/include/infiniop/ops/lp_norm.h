#ifndef __INFINIOP_LP_NORM_API_H__
#define __INFINIOP_LP_NORM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLPNormDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLPNormDescriptor(
    infiniopHandle_t handle,
    infiniopLPNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    int axis,
    int p,
    float eps);

__INFINI_C __export infiniStatus_t infiniopGetLPNormWorkspaceSize(infiniopLPNormDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLPNorm(infiniopLPNormDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *output,
                                                  const void *input,
                                                  void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLPNormDescriptor(infiniopLPNormDescriptor_t desc);

#endif
