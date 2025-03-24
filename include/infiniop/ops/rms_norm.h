#ifndef __INFINIOP_RMS_NORM_API_H__
#define __INFINIOP_RMS_NORM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRMSNormDescriptor_t;

__C __export infiniStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon);

__C __export infiniStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc, void *workspace, size_t workspace_size,
                                            void *y, const void *x, const void *w, void *stream);

__C __export infiniStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc);

#endif
