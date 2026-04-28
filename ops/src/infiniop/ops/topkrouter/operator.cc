#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/topkrouter.h"

#ifdef ENABLE_CPU_API
#include "cpu/topkrouter_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API) || defined(ENABLE_ALI_API)
#include "nvidia/topkrouter_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/topkrouter_metax.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/topkrouter_kunlun.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateTopkrouterDescriptor(infiniopHandle_t handle, infiniopTopkrouterDescriptor_t *desc_ptr,
                                                             infiniopTensorDescriptor_t x_desc,
                                                             infiniopTensorDescriptor_t correction_bias_desc) {
#define CREATE(CASE, NAMESPACE)                               \
    case CASE:                                                \
        return op::topkrouter::NAMESPACE::Descriptor::create( \
            handle, reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor **>(desc_ptr), x_desc, correction_bias_desc)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetTopkrouterWorkspaceSize(infiniopTopkrouterDescriptor_t desc, size_t *size) {
#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopTopkrouter(infiniopTopkrouterDescriptor_t desc, void *workspace, size_t workspace_size,
                                             void *values, void *indices, const void *x, const void *correction_bias,
                                             const float routed_scaling_factor, const size_t topk, void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                                          \
    case CASE:                                                                                                              \
        return reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor *>(desc)->calculate(                                  \
            workspace, workspace_size, (float *)values, (int *)indices, x, (float *)correction_bias, routed_scaling_factor, \
            topk, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyTopkrouterDescriptor(infiniopTopkrouterDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                \
    case CASE:                                                                  \
        delete reinterpret_cast<op::topkrouter::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
