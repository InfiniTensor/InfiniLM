#ifndef __INFINIOP_ONES_API_H__
#define __INFINIOP_ONES_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopOnesDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateOnesDescriptor(infiniopHandle_t handle,
                                                                infiniopOnesDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetOnesWorkspaceSize(infiniopOnesDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopOnes(infiniopOnesDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyOnesDescriptor(infiniopOnesDescriptor_t desc);

#endif
