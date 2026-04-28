#ifndef __INFINIOP_EMBEDDING_API_H__
#define __INFINIOP_EMBEDDING_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopEmbeddingDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateEmbeddingDescriptor(
    infiniopHandle_t handle,
    infiniopEmbeddingDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t weight_desc);

__INFINI_C __export infiniStatus_t infiniopEmbedding(
    infiniopEmbeddingDescriptor_t desc,
    void *output,
    const void *input,
    const void *weight,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyEmbeddingDescriptor(
    infiniopEmbeddingDescriptor_t desc);

#endif
