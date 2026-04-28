#ifndef __INFINIOP_VAR_API_H__
#define __INFINIOP_VAR_API_H__

#include "../operator_descriptor.h"
#include <cstddef>
#include <vector>
typedef struct InfiniopDescriptor *infiniopVarDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateVarDescriptor(infiniopHandle_t handle,
                                                               infiniopVarDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t var_output_desc,
                                                               infiniopTensorDescriptor_t input_desc,
                                                               size_t *dim,
                                                               size_t dim_size,
                                                               bool unbiased,
                                                               bool keepdim);

__INFINI_C __export infiniStatus_t infiniopGetVarWorkspaceSize(infiniopVarDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopVar(infiniopVarDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *var_output,
                                               const void *input,
                                               size_t *dim,
                                               size_t dim_size,
                                               bool unbiased,
                                               bool keepdim,
                                               void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyVarDescriptor(infiniopVarDescriptor_t desc);

#endif
