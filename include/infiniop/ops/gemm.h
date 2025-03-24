#ifndef __INFINIOP_GEMM_API_H__
#define __INFINIOP_GEMM_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopGemmDescriptor_t;

__C __export infiniStatus_t infiniopCreateGemmDescriptor(infiniopHandle_t handle,
                                                         infiniopGemmDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t c_desc,
                                                         infiniopTensorDescriptor_t a_desc,
                                                         infiniopTensorDescriptor_t b_desc);

__C __export infiniStatus_t infiniopGetGemmWorkspaceSize(infiniopGemmDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGemm(infiniopGemmDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *c,
                                         void const *a,
                                         void const *b,
                                         float alpha,
                                         float beta,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyGemmDescriptor(infiniopGemmDescriptor_t desc);

#endif
