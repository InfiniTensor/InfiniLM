#ifndef __INFINIOP_FLOOR_API_H__
#define __INFINIOP_FLOOR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopFloorDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateFloorDescriptor(infiniopHandle_t handle,
                                                                 infiniopFloorDescriptor_t *desc_ptr,
                                                                 infiniopTensorDescriptor_t output,
                                                                 infiniopTensorDescriptor_t intput);

__INFINI_C __export infiniStatus_t infiniopGetFloorWorkspaceSize(infiniopFloorDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopFloor(infiniopFloorDescriptor_t desc,
                                                 void *workspace,
                                                 size_t workspace_size,
                                                 void *output,
                                                 const void *intput,
                                                 void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyFloorDescriptor(infiniopFloorDescriptor_t desc);

#endif
