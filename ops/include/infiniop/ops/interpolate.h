#ifndef __INFINIOP_INTERPOLATE_API_H__
#define __INFINIOP_INTERPOLATE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopInterpolateDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateInterpolateDescriptor(infiniopHandle_t handle,
                                                                       infiniopInterpolateDescriptor_t *desc_ptr,
                                                                       infiniopTensorDescriptor_t y,
                                                                       infiniopTensorDescriptor_t x,
                                                                       const char *mode,
                                                                       void *size,
                                                                       void *scale_factor,
                                                                       int align_corners);

__INFINI_C __export infiniStatus_t infiniopGetInterpolateWorkspaceSize(infiniopInterpolateDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopInterpolate(infiniopInterpolateDescriptor_t desc,
                                                       void *workspace,
                                                       size_t workspace_size,
                                                       void *y,
                                                       const void *x,
                                                       void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyInterpolateDescriptor(infiniopInterpolateDescriptor_t desc);

#endif
