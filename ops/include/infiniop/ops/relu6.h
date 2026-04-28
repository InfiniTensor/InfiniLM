#ifndef __INFINIOP_RELU6_API_H__
#define __INFINIOP_RELU6_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopRelu6Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateRelu6Descriptor(infiniopHandle_t handle,
                                                                 infiniopRelu6Descriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetRelu6WorkspaceSize(infiniopRelu6Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopRelu6(infiniopRelu6Descriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *x,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyRelu6Descriptor(infiniopRelu6Descriptor_t desc);

#endif
