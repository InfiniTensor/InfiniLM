#ifndef __INFINIOP_FLOOR_DIVIDE_API_H__
#define __INFINIOP_FLOOR_DIVIDE_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFloorDivideDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFloorDivideDescriptor(infiniopHandle_t handle,
                                                                       infiniopFloorDivideDescriptor_t *desc_ptr,
                                                                       infiniopTensorDescriptor_t c,
                                                                       infiniopTensorDescriptor_t a,
                                                                       infiniopTensorDescriptor_t b);

__INFINI_C __export infiniStatus_t infiniopGetFloorDivideWorkspaceSize(infiniopFloorDivideDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopFloorDivide(infiniopFloorDivideDescriptor_t desc,
                                                       void *workspace,
                                                       size_t workspace_size,
                                                       void *c,
                                                       const void *a,
                                                       const void *b,
                                                       void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFloorDivideDescriptor(infiniopFloorDivideDescriptor_t desc);

#endif
