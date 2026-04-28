#ifndef __INFINIOP_ADAPTIVE_AVG_POOL1D_API_H__
#define __INFINIOP_ADAPTIVE_AVG_POOL1D_API_H__

#include "../operator_descriptor.h"

// 定义算子描述符类型
typedef struct InfiniopDescriptor *infiniopAdaptiveAvgPool1dDescriptor_t;

// 1. 创建算子描述符
__INFINI_C __export infiniStatus_t infiniopCreateAdaptiveAvgPool1dDescriptor(
    infiniopHandle_t handle,
    infiniopAdaptiveAvgPool1dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t input_desc);

// 2. 获取 Workspace 大小
__INFINI_C __export infiniStatus_t infiniopGetAdaptiveAvgPool1dWorkspaceSize(
    infiniopAdaptiveAvgPool1dDescriptor_t desc,
    size_t *size);

// 3. 执行计算
__INFINI_C __export infiniStatus_t infiniopAdaptiveAvgPool1d(
    infiniopAdaptiveAvgPool1dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    void *stream);

// 4. 销毁描述符
__INFINI_C __export infiniStatus_t infiniopDestroyAdaptiveAvgPool1dDescriptor(
    infiniopAdaptiveAvgPool1dDescriptor_t desc);

#endif // __INFINIOP_ADAPTIVE_AVG_POOL1D_API_H__
