#ifndef __INFINIOP_BITWISE_RIGHT_SHIFT_API_H__
#define __INFINIOP_BITWISE_RIGHT_SHIFT_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBitwiseRightShiftDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBitwiseRightShiftDescriptor(infiniopHandle_t handle,
                                                                             infiniopBitwiseRightShiftDescriptor_t *desc_ptr,
                                                                             infiniopTensorDescriptor_t y,
                                                                             infiniopTensorDescriptor_t x1,
                                                                             infiniopTensorDescriptor_t x2);

__INFINI_C __export infiniStatus_t infiniopGetBitwiseRightShiftWorkspaceSize(infiniopBitwiseRightShiftDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopBitwiseRightShift(infiniopBitwiseRightShiftDescriptor_t desc,
                                                             void *workspace,
                                                             size_t workspace_size,
                                                             void *y,
                                                             const void *x1,
                                                             const void *x2,
                                                             void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBitwiseRightShiftDescriptor(infiniopBitwiseRightShiftDescriptor_t desc);

#endif
