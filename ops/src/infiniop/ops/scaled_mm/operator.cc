#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/int8_gemm.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/int8_gemm_nvidia.cuh"
#endif

#if defined(ENABLE_MOORE_API)
#include "moore/int8_gemm_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateI8GemmDescriptor(infiniopHandle_t handle,
                                                         infiniopI8GemmDescriptor_t *desc_ptr,
                                                         infiniopTensorDescriptor_t out_desc,
                                                         infiniopTensorDescriptor_t bias_desc,
                                                         infiniopTensorDescriptor_t a_desc,
                                                         infiniopTensorDescriptor_t a_scale_desc,
                                                         infiniopTensorDescriptor_t b_desc,
                                                         infiniopTensorDescriptor_t b_scale_desc) {
#define CREATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        return op::i8gemm::NAMESPACE::Descriptor::create(                     \
            handle,                                                           \
            reinterpret_cast<op::i8gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            out_desc,                                                         \
            bias_desc,                                                        \
            a_desc,                                                           \
            a_scale_desc,                                                     \
            b_desc,                                                           \
            b_scale_desc);
    switch (handle->device) {
#if defined(ENABLE_NVIDIA_API)
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#if defined(ENABLE_QY_API)
        CREATE(INFINI_DEVICE_QY, nvidia)
#endif
#if defined(ENABLE_MOORE_API)
        CREATE(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetI8GemmWorkspaceSize(infiniopI8GemmDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                     \
    case CASE:                                                                                   \
        *size = reinterpret_cast<op::i8gemm::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize(); \
        return INFINI_STATUS_SUCCESS;
#if defined(ENABLE_NVIDIA_API)
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#if defined(ENABLE_QY_API)
        GET(INFINI_DEVICE_QY, nvidia)
#endif
#if defined(ENABLE_MOORE_API)
        GET(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopI8Gemm(infiniopI8GemmDescriptor_t desc,
                                         void *workspace,
                                         size_t workspace_size,
                                         void *out,
                                         const void *bias,
                                         const void *a,
                                         const void *a_scale,
                                         const void *b,
                                         const void *b_scale,
                                         void *stream) {
#define CACULATE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                         \
        return reinterpret_cast<op::i8gemm::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, out, bias, a, a_scale, b, b_scale, stream);
    switch (desc->device_type) {
#if defined(ENABLE_NVIDIA_API)
        CACULATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#if defined(ENABLE_QY_API)
        CACULATE(INFINI_DEVICE_QY, nvidia)
#endif
#if defined(ENABLE_MOORE_API)
        CACULATE(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CACULATE
}

__INFINI_C infiniStatus_t infiniopDestroyI8GemmDescriptor(infiniopI8GemmDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                            \
    case CASE:                                                              \
        delete reinterpret_cast<op::i8gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;
    switch (desc->device_type) {
#if defined(ENABLE_NVIDIA_API)
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#if defined(ENABLE_QY_API)
        DESTROY(INFINI_DEVICE_QY, nvidia)
#endif
#if defined(ENABLE_MOORE_API)
        DESTROY(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
