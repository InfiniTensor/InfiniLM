#ifndef __INFINIOP_PER_TENSOR_QUANT_INT8_API_H__
#define __INFINIOP_PER_TENSOR_QUANT_INT8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerTensorQuantI8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePerTensorQuantI8Descriptor(infiniopHandle_t handle,
                                                                            infiniopPerTensorQuantI8Descriptor_t *desc_ptr,
                                                                            infiniopTensorDescriptor_t x_packed_desc,
                                                                            infiniopTensorDescriptor_t x_scale_desc,
                                                                            infiniopTensorDescriptor_t x_zero_desc,
                                                                            infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetPerTensorQuantI8WorkspaceSize(infiniopPerTensorQuantI8Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPerTensorQuantI8(infiniopPerTensorQuantI8Descriptor_t desc,
                                                            void *workspace,
                                                            size_t workspace_size,
                                                            void *x_packed,
                                                            void *x_scale,
                                                            void *x_zero,
                                                            const void *x,
                                                            const bool is_static,
                                                            void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPerTensorQuantI8Descriptor(infiniopPerTensorQuantI8Descriptor_t desc);

#endif
