#ifndef __INFINIOP_I8GEMM_API_H__
#define __INFINIOP_I8GEMM_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopI8GemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateI8GemmDescriptor(infiniopHandle_t handle,
                                                                  infiniopI8GemmDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t out_desc,
                                                                  infiniopTensorDescriptor_t bias_desc,
                                                                  infiniopTensorDescriptor_t x_desc,
                                                                  infiniopTensorDescriptor_t x_scale_desc,
                                                                  infiniopTensorDescriptor_t weights_desc,
                                                                  infiniopTensorDescriptor_t weights_scale_desc);

__INFINI_C __export infiniStatus_t infiniopGetI8GemmWorkspaceSize(infiniopI8GemmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopI8Gemm(infiniopI8GemmDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *out,
                                                  const void *bias,
                                                  const void *x,
                                                  const void *x_scale,
                                                  const void *weights,
                                                  const void *weights_scale,
                                                  void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyI8GemmDescriptor(infiniopI8GemmDescriptor_t desc);

#endif
