#ifndef __INFINIOP_MATMUL_H__
#define __INFINIOP_MATMUL_H__

#include "../operator.h"

typedef InfiniopDescriptor *infiniopMatmulDescriptor_t;

__C __export infiniStatus_t infiniopCreateMatmulDescriptor(infiniopHandle_t handle,
                                                           infiniopMatmulDescriptor_t *desc_ptr,
                                                           infiniopTensorDescriptor_t c_desc,
                                                           infiniopTensorDescriptor_t a_desc,
                                                           infiniopTensorDescriptor_t b_desc);

__C __export infiniStatus_t infiniopGetMatmulWorkspaceSize(infiniopMatmulDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMatmul(infiniopMatmulDescriptor_t desc,
                                           void *workspace,
                                           size_t workspace_size,
                                           void *c,
                                           void const *a,
                                           void const *b,
                                           float alpha,
                                           float beta,
                                           void *stream);

__C __export infiniStatus_t infiniopDestroyMatmulDescriptor(infiniopMatmulDescriptor_t desc);

#endif
