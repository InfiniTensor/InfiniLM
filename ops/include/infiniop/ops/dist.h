#ifndef __INFINIOP_DIST_API_H__
#define __INFINIOP_DIST_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDistDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDistDescriptor(infiniopHandle_t handle,
                                                                infiniopDistDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x1,
                                                                infiniopTensorDescriptor_t x2,
                                                                double p);

__INFINI_C __export infiniStatus_t infiniopGetDistWorkspaceSize(infiniopDistDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDist(infiniopDistDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x1,
                                                const void *x2,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDistDescriptor(infiniopDistDescriptor_t desc);

#endif
