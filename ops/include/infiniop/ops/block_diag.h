#ifndef __INFINIOP_BLOCK_DIAG_API_H__
#define __INFINIOP_BLOCK_DIAG_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopBlockDiagDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateBlockDiagDescriptor(infiniopHandle_t handle,
                                                                     infiniopBlockDiagDescriptor_t *desc_ptr,
                                                                     infiniopTensorDescriptor_t y,
                                                                     infiniopTensorDescriptor_t *x,
                                                                     size_t num_inputs);

__INFINI_C __export infiniStatus_t infiniopGetBlockDiagWorkspaceSize(infiniopBlockDiagDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopBlockDiag(infiniopBlockDiagDescriptor_t desc,
                                                     void *workspace,
                                                     size_t workspace_size,
                                                     void *y,
                                                     const void **x,
                                                     void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyBlockDiagDescriptor(infiniopBlockDiagDescriptor_t desc);

#endif
