#ifndef __INFINIOP_HINGE_EMBEDDING_LOSS_API_H__
#define __INFINIOP_HINGE_EMBEDDING_LOSS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopHingeEmbeddingLossDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateHingeEmbeddingLossDescriptor(infiniopHandle_t handle,
                                                                              infiniopHingeEmbeddingLossDescriptor_t *desc_ptr,
                                                                              infiniopTensorDescriptor_t y,
                                                                              infiniopTensorDescriptor_t input,
                                                                              infiniopTensorDescriptor_t target,
                                                                              double margin,
                                                                              int reduction);

__INFINI_C __export infiniStatus_t infiniopGetHingeEmbeddingLossWorkspaceSize(infiniopHingeEmbeddingLossDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopHingeEmbeddingLoss(infiniopHingeEmbeddingLossDescriptor_t desc,
                                                              void *workspace,
                                                              size_t workspace_size,
                                                              void *y,
                                                              const void *input,
                                                              const void *target,
                                                              void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyHingeEmbeddingLossDescriptor(infiniopHingeEmbeddingLossDescriptor_t desc);

#endif
