#ifndef __INFINIOP_LOG10_API_H__
#define __INFINIOP_LOG10_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLog10Descriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLog10Descriptor(infiniopHandle_t handle,
                                                                 infiniopLog10Descriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t y,
                                                                 infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetLog10WorkspaceSize(infiniopLog10Descriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLog10(infiniopLog10Descriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *y,
                                                 const void *x,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLog10Descriptor(infiniopLog10Descriptor_t desc);

#endif
