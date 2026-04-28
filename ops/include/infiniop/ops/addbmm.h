#ifndef __INFINIOP_ADDBMM_API_H__
#define __INFINIOP_ADDBMM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAddbmmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAddbmmDescriptor(infiniopHandle_t handle,
                                                                  infiniopAddbmmDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t out_desc,
                                                                  infiniopTensorDescriptor_t input_desc,
                                                                  infiniopTensorDescriptor_t batch1_desc,
                                                                  infiniopTensorDescriptor_t batch2_desc,
                                                                  float alpha,
                                                                  float beta);

__INFINI_C __export infiniStatus_t infiniopGetAddbmmWorkspaceSize(infiniopAddbmmDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAddbmm(infiniopAddbmmDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *output,
                                                  const void *input,
                                                  const void *batch1,
                                                  const void *batch2,
                                                  void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAddbmmDescriptor(infiniopAddbmmDescriptor_t desc);

#endif // __INFINIOP_ADDBMM_API_H__
