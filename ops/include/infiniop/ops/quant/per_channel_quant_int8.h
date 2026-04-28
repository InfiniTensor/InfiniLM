#ifndef __INFINIOP_PER_CHANNEL_QUANT_INT8_API_H__
#define __INFINIOP_PER_CHANNEL_QUANT_INT8_API_H__

#include "../../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopPerChannelQuantI8Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreatePerChannelQuantI8Descriptor(infiniopHandle_t handle,
                                                                             infiniopPerChannelQuantI8Descriptor_t *desc_ptr,
                                                                             infiniopTensorDescriptor_t x_packed_desc,
                                                                             infiniopTensorDescriptor_t x_scale_desc,
                                                                             infiniopTensorDescriptor_t x_zero_desc,
                                                                             infiniopTensorDescriptor_t x_desc);

__INFINI_C __export infiniStatus_t infiniopGetPerChannelQuantI8WorkspaceSize(infiniopPerChannelQuantI8Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopPerChannelQuantI8(infiniopPerChannelQuantI8Descriptor_t desc,
                                                             void *workspace,
                                                             size_t workspace_size,
                                                             void *x_packed,
                                                             void *x_scale,
                                                             void *x_zero,
                                                             const void *x,
                                                             void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyPerChannelQuantI8Descriptor(infiniopPerChannelQuantI8Descriptor_t desc);

#endif
