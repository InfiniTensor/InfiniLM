#ifndef __INFINIOP_LERP_API_H__
#define __INFINIOP_LERP_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopLerpDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateLerpDescriptor(infiniopHandle_t handle,
                                                                infiniopLerpDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t output,
                                                                infiniopTensorDescriptor_t start,
                                                                infiniopTensorDescriptor_t end,
                                                                infiniopTensorDescriptor_t weight,
                                                                float weight_scalar);

__INFINI_C __export infiniStatus_t infiniopGetLerpWorkspaceSize(infiniopLerpDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopLerp(infiniopLerpDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *output,
                                                const void *start,
                                                const void *end,
                                                const void *weight,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyLerpDescriptor(infiniopLerpDescriptor_t desc);

#endif // __INFINIOP_LERP_API_H__
