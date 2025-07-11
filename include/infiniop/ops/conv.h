#ifndef __INFINIOP_CONV_API_H__
#define __INFINIOP_CONV_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopConvDescriptor_t;

__C __export infiniStatus_t infiniopCreateConvDescriptor(infiniopHandle_t handle,
                                                         infiniopConvDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y_desc,
                                                         infiniopTensorDescriptor_t x_desc,
                                                         infiniopTensorDescriptor_t w_desc,
                                                         infiniopTensorDescriptor_t b_desc,
                                                         void *pads,
                                                         void *strides,
                                                         void *dilations,
                                                         size_t n);

__C __export infiniStatus_t infiniopGetConvWorkspaceSize(infiniopConvDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopConv(infiniopConvDescriptor_t desc, void *workspace, size_t workspace_size, void *y, const void *x, const void *w, const void *bias, void *stream);

__C __export infiniStatus_t infiniopDestroyConvDescriptor(infiniopConvDescriptor_t desc);

#endif
