#ifndef __INFINIOP_DIFF_API_H__
#define __INFINIOP_DIFF_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDiffDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDiffDescriptor(infiniopHandle_t handle,
                                                                infiniopDiffDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x,
                                                                int dim,
                                                                int n);

__INFINI_C __export infiniStatus_t infiniopGetDiffWorkspaceSize(infiniopDiffDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDiff(infiniopDiffDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDiffDescriptor(infiniopDiffDescriptor_t desc);

#endif
