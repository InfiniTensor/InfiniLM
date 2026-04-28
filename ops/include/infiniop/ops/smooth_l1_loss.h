#ifndef __INFINIOP_SMOOTH_L1_LOSS_API_H__
#define __INFINIOP_SMOOTH_L1_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopSmoothL1LossDescriptor_t;
__INFINI_C __export infiniStatus_t infiniopCreateSmoothL1LossDescriptor(infiniopHandle_t handle,
                                                                        infiniopSmoothL1LossDescriptor_t *desc_ptr,
                                                                        infiniopTensorDescriptor_t output,
                                                                        infiniopTensorDescriptor_t input,
                                                                        infiniopTensorDescriptor_t target,
                                                                        float beta,
                                                                        int reduction);

__INFINI_C __export infiniStatus_t infiniopGetSmoothL1LossWorkspaceSize(infiniopSmoothL1LossDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSmoothL1Loss(infiniopSmoothL1LossDescriptor_t desc,
                                                        void *workspace,
                                                        size_t workspace_size,
                                                        void *output,
                                                        const void *input,
                                                        const void *target,
                                                        void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySmoothL1LossDescriptor(infiniopSmoothL1LossDescriptor_t desc);

#endif // __INFINIOP_SMOOTH_L1_LOSS_API_H__
