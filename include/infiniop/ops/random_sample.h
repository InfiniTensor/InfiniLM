#ifndef __INFINIOP_RANDOM_SAMPLE_API_H__
#define __INFINIOP_RANDOM_SAMPLE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRandomSampleDescriptor_t;

__C __export infiniStatus_t infiniopCreateRandomSampleDescriptor(
    infiniopHandle_t handle,
    infiniopRandomSampleDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t result,
    infiniopTensorDescriptor_t probs);

__C __export infiniStatus_t infiniopGetRandomSampleWorkspaceSize(
    infiniopRandomSampleDescriptor_t desc,
    size_t *size);

__C __export infiniStatus_t infiniopRandomSample(
    infiniopRandomSampleDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *result,
    const void *probs,
    float random_val,
    float topp,
    int topk,
    float temperature,
    void *stream);

__C __export infiniStatus_t infiniopDestroyRandomSampleDescriptor(
    infiniopRandomSampleDescriptor_t desc);

#endif
