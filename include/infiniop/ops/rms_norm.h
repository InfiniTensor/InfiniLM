#ifndef __INFINIOP_RMS_NORM_H__
#define __INFINIOP_RMS_NORM_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopRMSNormDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon);

__C __export infiniopStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc, size_t *size);

__C __export infiniopStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc, void *workspace, size_t workspace_size,
                                              void *y, void const *x, void const *w, void *stream);

__C __export infiniopStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc);

#endif
