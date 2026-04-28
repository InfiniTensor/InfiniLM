#ifndef __INFINIOP_DEQUANTIZE_GPTQ_API_H__
#define __INFINIOP_DEQUANTIZE_GPTQ_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDequantizeGPTQDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDequantizeGPTQDescriptor(infiniopHandle_t handle,
                                                                          infiniopDequantizeGPTQDescriptor_t *desc_ptr,
                                                                          infiniopTensorDescriptor_t out_desc,
                                                                          infiniopTensorDescriptor_t qweight_desc,
                                                                          infiniopTensorDescriptor_t scales_desc,
                                                                          infiniopTensorDescriptor_t zeros_desc,
                                                                          infiniopTensorDescriptor_t g_idx_desc); // add g_idx

__INFINI_C __export infiniStatus_t infiniopGetDequantizeGPTQWorkspaceSize(infiniopDequantizeGPTQDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDequantizeGPTQ(infiniopDequantizeGPTQDescriptor_t desc,
                                                          void *workspace,
                                                          size_t workspace_size,
                                                          void *out,
                                                          const void *qweight,
                                                          const void *scales,
                                                          const void *zeros,
                                                          const void *g_idx, // add g_idx
                                                          void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDequantizeGPTQDescriptor(infiniopDequantizeGPTQDescriptor_t desc);

#endif
