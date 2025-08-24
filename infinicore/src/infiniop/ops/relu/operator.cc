#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/relu.h"

#ifdef ENABLE_CPU_API
#include "cpu/relu_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#ifdef ENABLE_NINETOOTHED
#include "nvidia/relu_nvidia.cuh"
#endif
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
#include "metax/relu_metax.h"
#endif
#endif

__C infiniStatus_t infiniopCreateReluDescriptor(
    infiniopHandle_t handle,
    infiniopReluDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::relu::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::relu::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                         \
            {x_desc})

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetReluWorkspaceSize(infiniopReluDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                              \
        *size = reinterpret_cast<op::relu::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_ILUVATAR, nvidia)
#endif
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        GET(INFINI_DEVICE_METAX, metax)
#endif
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRelu(
    infiniopReluDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::relu::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, {x}, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyReluDescriptor(infiniopReluDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::relu::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#endif
#ifdef ENABLE_ILUVATAR_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#endif
#ifdef ENABLE_METAX_API
#ifdef ENABLE_NINETOOTHED
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
