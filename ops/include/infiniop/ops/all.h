#ifndef __INFINIOP_ALL_API_H__
#define __INFINIOP_ALL_API_H__

#include "../operator_descriptor.h"
#include <cstddef>
#include <vector>
typedef struct InfiniopDescriptor *infiniopAllDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAllDescriptor(infiniopHandle_t handle,
                                                               infiniopAllDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t output_desc,
                                                               infiniopTensorDescriptor_t input_desc,
                                                               size_t *dim,
                                                               size_t dim_size,
                                                               bool keepdim);

__INFINI_C __export infiniStatus_t infiniopGetAllWorkspaceSize(infiniopAllDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAll(infiniopAllDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *output,
                                               const void *input,
                                               size_t *dim,
                                               size_t dim_size,
                                               bool keepdim,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAllDescriptor(infiniopAllDescriptor_t desc);

#endif
