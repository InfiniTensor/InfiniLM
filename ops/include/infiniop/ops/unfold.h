#ifndef __INFINIOP_UNFOLD_API_H__
#define __INFINIOP_UNFOLD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopUnfoldDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateUnfoldDescriptor(infiniopHandle_t handle,
                                                                  infiniopUnfoldDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t output,
                                                                  infiniopTensorDescriptor_t input,
                                                                  const int *kernel_sizes,
                                                                  const int *strides,
                                                                  const int *paddings,
                                                                  const int *dilations);

// 获取 Unfold 工作区大小
__INFINI_C __export infiniStatus_t infiniopGetUnfoldWorkspaceSize(infiniopUnfoldDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopUnfold(infiniopUnfoldDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *output,
                                                  const void *input,
                                                  void *stream);
__INFINI_C __export infiniStatus_t infiniopDestroyUnfoldDescriptor(infiniopUnfoldDescriptor_t desc);

#endif // __INFINIOP_UNFOLD_API_H__
