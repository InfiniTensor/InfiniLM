#ifndef __INFINIOP_MULTI_MARGIN_LOSS_API_H__
#define __INFINIOP_MULTI_MARGIN_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMultiMarginLossDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateMultiMarginLossDescriptor(infiniopHandle_t handle,
                                                                           infiniopMultiMarginLossDescriptor_t *desc_ptr,
                                                                           infiniopTensorDescriptor_t output,
                                                                           infiniopTensorDescriptor_t input,
                                                                           infiniopTensorDescriptor_t target,
                                                                           infiniopTensorDescriptor_t weight,
                                                                           int p,
                                                                           float margin,
                                                                           int reduction);

__INFINI_C __export infiniStatus_t infiniopGetMultiMarginLossWorkspaceSize(infiniopMultiMarginLossDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopMultiMarginLoss(infiniopMultiMarginLossDescriptor_t desc,
                                                           void *workspace,
                                                           size_t workspace_size,
                                                           void *output,
                                                           const void *input,
                                                           const void *target,
                                                           const void *weight,
                                                           void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyMultiMarginLossDescriptor(infiniopMultiMarginLossDescriptor_t desc);

#endif // __INFINIOP_MULTI_MARGIN_LOSS_API_H__
