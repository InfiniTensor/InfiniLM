#ifndef __INFINIOP_TOPKRouter_API_H__
#define __INFINIOP_TOPKRouter_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTopkrouterDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTopkrouterDescriptor(infiniopHandle_t handle,
                                                                      infiniopTopkrouterDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t x_desc,
                                                                      infiniopTensorDescriptor_t correction_bias_desc);

__INFINI_C __export infiniStatus_t infiniopGetTopkrouterWorkspaceSize(infiniopTopkrouterDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTopkrouter(infiniopTopkrouterDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *values,
                                                      void *indices,
                                                      const void *x,
                                                      const void *correction_bias,
                                                      const float routed_scaling_factor,
                                                      const size_t topk,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTopkrouterDescriptor(infiniopTopkrouterDescriptor_t desc);

#endif
