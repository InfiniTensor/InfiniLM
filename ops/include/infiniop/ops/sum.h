#ifndef __INFINIOP_SUM_API_H__
#define __INFINIOP_SUM_API_H__

#include "../operator_descriptor.h"
#include <cstddef>
#include <vector>
typedef struct InfiniopDescriptor *infiniopSumDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateSumDescriptor(infiniopHandle_t handle,
                                                               infiniopSumDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t output_desc,
                                                               infiniopTensorDescriptor_t input_desc,
                                                               size_t *dim,
                                                               size_t dim_size,
                                                               bool keepdim);

__INFINI_C __export infiniStatus_t infiniopGetSumWorkspaceSize(infiniopSumDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopSum(infiniopSumDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *output,
                                               const void *input,
                                               size_t *dim,
                                               size_t dim_size,
                                               bool keepdim,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroySumDescriptor(infiniopSumDescriptor_t desc);

#endif
