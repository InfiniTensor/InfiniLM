#ifndef __INFINIOP_LOG1P_API_H__
#define __INFINIOP_LOG1P_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLog1pDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLog1pDescriptor(infiniopHandle_t handle,
                                                                 infiniopLog1pDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetLog1pWorkspaceSize(infiniopLog1pDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLog1p(infiniopLog1pDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *x,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLog1pDescriptor(infiniopLog1pDescriptor_t desc);

#endif
