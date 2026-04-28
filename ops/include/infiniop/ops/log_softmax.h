#ifndef __INFINIOP_LOG_SOFTMAX_API_H__
#define __INFINIOP_LOG_SOFTMAX_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLogSoftmaxDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLogSoftmaxDescriptor(infiniopHandle_t handle,
                                                                      infiniopLogSoftmaxDescriptor_t *desc_ptr,
                                                                      infiniopTensorDescriptor_t output,
                                                                      infiniopTensorDescriptor_t input,
                                                                      int dim);

__INFINI_C __export infiniStatus_t infiniopGetLogSoftmaxWorkspaceSize(infiniopLogSoftmaxDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLogSoftmax(infiniopLogSoftmaxDescriptor_t desc,
                                                      void *workspace,
                                                      size_t workspace_size,
                                                      void *output,
                                                      const void *input,
                                                      void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLogSoftmaxDescriptor(infiniopLogSoftmaxDescriptor_t desc);

#endif // __INFINIOP_LOG_SOFTMAX_API_H__
