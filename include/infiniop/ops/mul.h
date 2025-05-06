#ifndef __INFINIOP_MUL_API_H__
#define __INFINIOP_MUL_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopMulDescriptor_t;

__C __export infiniStatus_t infiniopCreateMulDescriptor(infiniopHandle_t handle,
                                                       infiniopMulDescriptor_t *desc_ptr,
                                                       infiniopTensorDescriptor_t c,
                                                       infiniopTensorDescriptor_t a,
                                                       infiniopTensorDescriptor_t b);

__C __export infiniStatus_t infiniopGetMulWorkspaceSize(infiniopMulDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMul(infiniopMulDescriptor_t desc,
                                        void *workspace,
                                        size_t workspace_size,
                                        void *c,
                                        const void *a,
                                        const void *b,
                                        void *stream);
                
__C __export infiniStatus_t infiniopDestroyMulDescriptor(infiniopMulDescriptor_t desc);

#endif
