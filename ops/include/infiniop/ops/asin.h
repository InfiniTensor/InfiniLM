#ifndef __INFINIOP_ASIN_API_H__
#define __INFINIOP_ASIN_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAsinDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAsinDescriptor(infiniopHandle_t handle,
                                                                infiniopAsinDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t output,
                                                                infiniopTensorDescriptor_t input);

__INFINI_C __export infiniStatus_t infiniopGetAsinWorkspaceSize(infiniopAsinDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAsin(infiniopAsinDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *output,
                                                const void *input,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAsinDescriptor(infiniopAsinDescriptor_t desc);

#endif
