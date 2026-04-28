#ifndef __INFINIOP_TOPK_API_H__
#define __INFINIOP_TOPK_API_H__

#include "../operator_descriptor.h"
#include <cstddef>
#include <vector>
typedef struct InfiniopDescriptor *infiniopTopKDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateTopKDescriptor(infiniopHandle_t handle,
                                                                infiniopTopKDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t values_output_desc,
                                                                infiniopTensorDescriptor_t indices_output_desc,
                                                                infiniopTensorDescriptor_t input_desc,
                                                                size_t k,
                                                                size_t dim,
                                                                bool largest,
                                                                bool sorted);

__INFINI_C __export infiniStatus_t infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopTopK(infiniopTopKDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *values_output,
                                                void *indices_output,
                                                const void *input,
                                                size_t k,
                                                size_t dim,
                                                bool largest,
                                                bool sorted,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc);

#endif
