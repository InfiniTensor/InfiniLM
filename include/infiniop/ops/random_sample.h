#ifndef __INFINIOP_RANDOM_SAMPLE_H__
#define __INFINIOP_RANDOM_SAMPLE_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopRandomSampleDescriptor_t;

__C __export infiniopStatus_t infiniopCreateRandomSampleDescriptor(infiniopHandle_t handle, infiniopRandomSampleDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result, infiniopTensorDescriptor_t probs);

__C __export infiniopStatus_t infiniopGetRandomSampleWorkspaceSize(infiniopRandomSampleDescriptor_t desc, size_t *size);

__C __export infiniopStatus_t infiniopRandomSample(infiniopRandomSampleDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   void *result,
                                                   void const *probs,
                                                   float random_val,
                                                   float topp,
                                                   int topk,
                                                   float temperature,
                                                   void *stream);

__C __export infiniopStatus_t infiniopDestroyRandomSampleDescriptor(infiniopRandomSampleDescriptor_t desc);


#endif
