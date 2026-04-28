#ifndef __INFINIOP_TRIPLET_MARGIN_WITH_DISTANCE_LOSS_API_H__
#define __INFINIOP_TRIPLET_MARGIN_WITH_DISTANCE_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTripletMarginWithDistanceLossDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTripletMarginWithDistanceLossDescriptor(
    infiniopHandle_t handle,
    infiniopTripletMarginWithDistanceLossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t anchor,
    infiniopTensorDescriptor_t positive,
    infiniopTensorDescriptor_t negative,
    float margin,
    int swap,
    int reduction);
__INFINI_C __export infiniStatus_t infiniopGetTripletMarginWithDistanceLossWorkspaceSize(
    infiniopTripletMarginWithDistanceLossDescriptor_t desc,
    size_t *size);
__INFINI_C __export infiniStatus_t infiniopTripletMarginWithDistanceLoss(infiniopTripletMarginWithDistanceLossDescriptor_t desc,
                                                                         void *workspace,
                                                                         size_t workspace_size,
                                                                         void *output,
                                                                         const void *anchor,
                                                                         const void *positive,
                                                                         const void *negative,
                                                                         void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTripletMarginWithDistanceLossDescriptor(
    infiniopTripletMarginWithDistanceLossDescriptor_t desc);
#endif // __INFINIOP_TRIPLET_MARGIN_WITH_DISTANCE_LOSS_API_H__
