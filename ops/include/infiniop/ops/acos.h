#ifndef __INFINIOP_ACOS_API_H__
#define __INFINIOP_ACOS_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAcosDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAcosDescriptor(infiniopHandle_t handle,
                                                                infiniopAcosDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t y,
                                                                infiniopTensorDescriptor_t x);

__INFINI_C __export infiniStatus_t infiniopGetAcosWorkspaceSize(infiniopAcosDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAcos(infiniopAcosDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *y,
                                                const void *x,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAcosDescriptor(infiniopAcosDescriptor_t desc);

#endif
