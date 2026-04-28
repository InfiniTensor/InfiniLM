#ifndef __INFINIOP_PER_TENSOR_DEQUANT_INT8_API_H__
#define __INFINIOP_PER_TENSOR_DEQUANT_INT8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerTensorDequantI8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePerTensorDequantI8Descriptor(infiniopHandle_t handle,
                                                                              infiniopPerTensorDequantI8Descriptor_t *desc_ptr,
                                                                              infiniopTensorDescriptor_t x_desc,
                                                                              infiniopTensorDescriptor_t x_packed_desc,
                                                                              infiniopTensorDescriptor_t x_scale_desc,
                                                                              infiniopTensorDescriptor_t x_zero_desc);

__INFINI_C __export infiniStatus_t infiniopGetPerTensorDequantI8WorkspaceSize(infiniopPerTensorDequantI8Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPerTensorDequantI8(infiniopPerTensorDequantI8Descriptor_t desc,
                                                              void *workspace,
                                                              size_t workspace_size,
                                                              void *x,
                                                              const void *x_packed,
                                                              const void *x_scale,
                                                              const void *x_zero,
                                                              void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPerTensorDequantI8Descriptor(infiniopPerTensorDequantI8Descriptor_t desc);

#endif
