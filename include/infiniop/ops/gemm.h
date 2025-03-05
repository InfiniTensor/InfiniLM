#ifndef __INFINIOP_GEMM_API_H__
#define __INFINIOP_GEMM_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopGEMMDescriptor_t;

__C __export infiniStatus_t infiniopCreateGEMMDescriptor(infiniopHandle_t handle,
                                                         infiniopGEMMDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t y_desc,
                                                         infiniopTensorDescriptor_t a_desc,
                                                         infiniopTensorDescriptor_t b_desc,
                                                         infiniopTensorDescriptor_t c_desc,
                                                         char transA,
                                                         char transB);

__C __export infiniStatus_t infiniopGetGEMMWorkspaceSize(infiniopGEMMDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopGEMM(infiniopGEMMDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *y,
                                         void const *a,
                                         void const *b,
                                         void const *c,
                                         float alpha,
                                         float beta,
                                         void *stream);

__C __export infiniStatus_t infiniopDestroyGEMMDescriptor(infiniopGEMMDescriptor_t desc);
#endif
