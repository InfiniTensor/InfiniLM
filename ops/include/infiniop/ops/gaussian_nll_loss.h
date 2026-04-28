#ifndef __INFINIOP_GAUSSIAN_NLL_LOSS_API_H__
#define __INFINIOP_GAUSSIAN_NLL_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGaussianNllLossDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGaussianNllLossDescriptor(infiniopHandle_t handle,
                                                                           infiniopGaussianNllLossDescriptor_t *desc_ptr,
                                                                           infiniopTensorDescriptor_t y,
                                                                           infiniopTensorDescriptor_t input,
                                                                           infiniopTensorDescriptor_t target,
                                                                           infiniopTensorDescriptor_t var,
                                                                           int full,
                                                                           double eps,
                                                                           int reduction);

__INFINI_C __export infiniStatus_t infiniopGetGaussianNllLossWorkspaceSize(infiniopGaussianNllLossDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopGaussianNllLoss(infiniopGaussianNllLossDescriptor_t desc,
                                                           void *workspace,
                                                           size_t workspace_size,
                                                           void *y,
                                                           const void *input,
                                                           const void *target,
                                                           const void *var,
                                                           void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGaussianNllLossDescriptor(infiniopGaussianNllLossDescriptor_t desc);

#endif
