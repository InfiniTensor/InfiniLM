#ifndef __INFINIOP_HUBER_LOSS_API_H__
#define __INFINIOP_HUBER_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopHuberLossDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHuberLossDescriptor(infiniopHandle_t handle,
                                                                     infiniopHuberLossDescriptor_t *desc_ptr,
                                                                     infiniopTensorDescriptor_t output,
                                                                     infiniopTensorDescriptor_t input,
                                                                     infiniopTensorDescriptor_t target,
                                                                     float delta,
                                                                     int reduction);

__INFINI_C __export infiniStatus_t infiniopGetHuberLossWorkspaceSize(infiniopHuberLossDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHuberLoss(infiniopHuberLossDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *output,
                                                     const void *input,
                                                     const void *target,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHuberLossDescriptor(infiniopHuberLossDescriptor_t desc);

#endif // __INFINIOP_HUBER_LOSS_API_H__
