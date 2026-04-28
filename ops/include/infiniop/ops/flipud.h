#ifndef __INFINIOP_FLIPUD_API_H__
#define __INFINIOP_FLIPUD_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFlipudDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFlipudDescriptor(infiniopHandle_t handle,
                                                                  infiniopFlipudDescriptor_t *desc_ptr,
                                                                  infiniopTensorDescriptor_t output,
                                                                  infiniopTensorDescriptor_t input);

// 获取工作空间大小
__INFINI_C __export infiniStatus_t infiniopGetFlipudWorkspaceSize(infiniopFlipudDescriptor_t desc, size_t *size);

// 执行 Flipud 算子
__INFINI_C __export infiniStatus_t infiniopFlipud(infiniopFlipudDescriptor_t desc,
                                                  void *workspace,
                                                  size_t workspace_size,
                                                  void *output,
                                                  const void *input,
                                                  void *stream);

// 销毁描述符
__INFINI_C __export infiniStatus_t infiniopDestroyFlipudDescriptor(infiniopFlipudDescriptor_t desc);

#endif // __INFINIOP_FLIPUD_API_H__
