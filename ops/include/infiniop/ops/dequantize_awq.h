#ifndef __INFINIOP_DEQUANTIZE_AWQ_API_H__
#define __INFINIOP_DEQUANTIZE_AWQ_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopDequantizeAWQDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateDequantizeAWQDescriptor(infiniopHandle_t handle,
                                                                         infiniopDequantizeAWQDescriptor_t *desc_ptr,
                                                                         infiniopTensorDescriptor_t out_desc,
                                                                         infiniopTensorDescriptor_t qweight_desc,
                                                                         infiniopTensorDescriptor_t scales_desc,
                                                                         infiniopTensorDescriptor_t zeros_desc);

__INFINI_C __export infiniStatus_t infiniopGetDequantizeAWQWorkspaceSize(infiniopDequantizeAWQDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopDequantizeAWQ(infiniopDequantizeAWQDescriptor_t desc,
                                                         void *workspace,
                                                         size_t workspace_size,
                                                         void *out,
                                                         const void *qweight,
                                                         const void *scales,
                                                         const void *zeros,
                                                         void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyDequantizeAWQDescriptor(infiniopDequantizeAWQDescriptor_t desc);

#endif
