#ifndef __INFINIOP_GPTQ_QYBLAS_GEMM_API_H__
#define __INFINIOP_GPTQ_QYBLAS_GEMM_API_H__

#include "../operator_descriptor.h"
#include <cstdint>

typedef struct InfiniopDescriptor *infiniopGptqQyblasGemmDescriptor_t;

__INFINI_C __export infiniStatus_t infiniopCreateGptqQyblasGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGptqQyblasGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc);

__INFINI_C __export infiniStatus_t infiniopGetGptqQyblasGemmWorkspaceSize(
    infiniopGptqQyblasGemmDescriptor_t desc,
    size_t *size);

__INFINI_C __export infiniStatus_t infiniopGptqQyblasGemm(
    infiniopGptqQyblasGemmDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    void *b_scale,
    void *b_zero,
    int64_t quant_type,
    int64_t bit,
    void *stream);

__INFINI_C __export infiniStatus_t infiniopDestroyGptqQyblasGemmDescriptor(
    infiniopGptqQyblasGemmDescriptor_t desc);
#endif
