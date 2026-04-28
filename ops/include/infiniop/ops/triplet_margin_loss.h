#ifndef __INFINIOP_TRIPLET_MARGIN_LOSS_API_H__
#define __INFINIOP_TRIPLET_MARGIN_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTripletMarginLossDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTripletMarginLossDescriptor(infiniopHandle_t handle,
                                                                             infiniopTripletMarginLossDescriptor_t *desc_ptr,
                                                                             infiniopTensorDescriptor_t output,
                                                                             infiniopTensorDescriptor_t anchor,
                                                                             infiniopTensorDescriptor_t positive,
                                                                             infiniopTensorDescriptor_t negative,
                                                                             float margin,
                                                                             int p,
                                                                             float eps,
                                                                             int swap,       // 0: False, 1: True
                                                                             int reduction); // 0: None, 1: Mean, 2: Sum

__INFINI_C __export infiniStatus_t infiniopGetTripletMarginLossWorkspaceSize(infiniopTripletMarginLossDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTripletMarginLoss(infiniopTripletMarginLossDescriptor_t desc,
                                                             void *workspace,
                                                             size_t workspace_size,
                                                             void *output,
                                                             const void *anchor,
                                                             const void *positive,
                                                             const void *negative,
                                                             void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTripletMarginLossDescriptor(infiniopTripletMarginLossDescriptor_t desc);

#endif // __INFINIOP_TRIPLET_MARGIN_LOSS_API_H__
