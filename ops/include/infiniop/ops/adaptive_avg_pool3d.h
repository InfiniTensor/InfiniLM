#ifndef INFINIOP_ADAPTIVE_AVG_POOL3D_H_
#define INFINIOP_ADAPTIVE_AVG_POOL3D_H_

#include "../operator_descriptor.h"
#include <cstddef>

typedef struct InfiniopDescriptor *infiniopAdaptiveAvgPool3DDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAdaptiveAvgPool3DDescriptor(
    infiniopHandle_t handle,
    infiniopAdaptiveAvgPool3DDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    size_t *output_size);

__INFINI_C __export infiniStatus_t infiniopGetAdaptiveAvgPool3DWorkspaceSize(
    infiniopAdaptiveAvgPool3DDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopAdaptiveAvgPool3D(
    infiniopAdaptiveAvgPool3DDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAdaptiveAvgPool3DDescriptor(infiniopAdaptiveAvgPool3DDescriptor_t desc);

#endif
