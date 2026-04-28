#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gptq_qyblas_gemm.h"

#if defined ENABLE_QY_API
#include "nvidia/gptq_qyblas_gemm_nvidia.cuh"
#endif

__INFINI_C infiniStatus_t infiniopCreateGptqQyblasGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGptqQyblasGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scales_desc,
    infiniopTensorDescriptor_t b_zeros_desc) {

#define CREATE(CASE, NAMESPACE)                                                         \
    case CASE:                                                                          \
        return op::gptq_qyblas_gemm::NAMESPACE::Descriptor::create(                     \
            handle,                                                                     \
            reinterpret_cast<op::gptq_qyblas_gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc, a_desc, b_desc, b_scales_desc, b_zeros_desc);

    switch (handle->device) {

#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia)
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetGptqQyblasGemmWorkspaceSize(
    infiniopGptqQyblasGemmDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                            \
    case CASE:                                                                                          \
        *size = reinterpret_cast<op::gptq_qyblas_gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopGptqQyblasGemm(
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
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                   \
        return reinterpret_cast<op::gptq_qyblas_gemm::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, a, b, b_scale, b_zero, quant_type, bit, stream);

    switch (desc->device_type) {

#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyGptqQyblasGemmDescriptor(
    infiniopGptqQyblasGemmDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                      \
    case CASE:                                                                        \
        delete reinterpret_cast<op::gptq_qyblas_gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}
