#ifndef __INFINIOP_ADD_RMS_NORM_API_H__
#define __INFINIOP_ADD_RMS_NORM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAddRMSNormDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAddRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopAddRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t residual_out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t weight_desc,
    float epsilon);

__INFINI_C __export infiniStatus_t infiniopGetAddRMSNormWorkspaceSize(infiniopAddRMSNormDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAddRMSNorm(infiniopAddRMSNormDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *y,
                                                      void *residual_out,
                                                      const void *a,
                                                      const void *b,
                                                      const void *weight,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAddRMSNormDescriptor(infiniopAddRMSNormDescriptor_t desc);

#endif
