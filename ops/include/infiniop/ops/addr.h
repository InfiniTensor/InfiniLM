#ifndef __INFINIOP_ADDR_API_H__
#define __INFINIOP_ADDR_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopAddrDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateAddrDescriptor(infiniopHandle_t handle,
                                                                infiniopAddrDescriptor_t *desc_ptr,
                                                                infiniopTensorDescriptor_t out,
                                                                infiniopTensorDescriptor_t input,
                                                                infiniopTensorDescriptor_t vec1,
                                                                infiniopTensorDescriptor_t vec2,
                                                                float beta,
                                                                float alpha);

__INFINI_C __export infiniStatus_t infiniopGetAddrWorkspaceSize(infiniopAddrDescriptor_t desc, size_t *size);

__INFINI_C __export infiniStatus_t infiniopAddr(infiniopAddrDescriptor_t desc,
                                                void *workspace,
                                                size_t workspace_size,
                                                void *out,
                                                const void *input,
                                                const void *vec1,
                                                const void *vec2,
                                                void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyAddrDescriptor(infiniopAddrDescriptor_t desc);

#endif
