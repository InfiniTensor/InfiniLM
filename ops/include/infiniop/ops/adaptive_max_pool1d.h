#ifndef __INFINIOP_ADAPTIVE_MAX_POOL1D_H__
#define __INFINIOP_ADAPTIVE_MAX_POOL1D_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAdaptiveMaxPool1dDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAdaptiveMaxPool1dDescriptor(
    infiniopHandle_t handle,
    infiniopAdaptiveMaxPool1dDescriptor_t *desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t output_size);

__INFINI_C __export infiniStatus_t infiniopGetAdaptiveMaxPool1dWorkspaceSize(infiniopAdaptiveMaxPool1dDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAdaptiveMaxPool1d(infiniopAdaptiveMaxPool1dDescriptor_t desc, void *workspace, size_t workspace_size,
                                                             void *y, const void *x, void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAdaptiveMaxPool1dDescriptor(infiniopAdaptiveMaxPool1dDescriptor_t desc);

#endif
