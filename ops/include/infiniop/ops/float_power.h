#ifndef __INFINIOP_FLOAT_POWER_API_H__
#define __INFINIOP_FLOAT_POWER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFloatPowerDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFloatPowerDescriptor(infiniopHandle_t handle,
                                                                      infiniopFloatPowerDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t y,
                                                                      infiniopTensorDescriptor_t x,
                                                                      infiniopTensorDescriptor_t exponent,
                                                                      float scalar_exponent);

__INFINI_C __export infiniStatus_t infiniopGetFloatPowerWorkspaceSize(infiniopFloatPowerDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopFloatPower(infiniopFloatPowerDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *y,
                                                      const void *x,
                                                      const void *exponent,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFloatPowerDescriptor(infiniopFloatPowerDescriptor_t desc);

#endif
