#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/adaptive_max_pool1d.h"

#ifdef ENABLE_CPU_API
#include "cpu/adaptive_max_pool1d_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/adaptive_max_pool1d_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/adaptive_max_pool1d_metax.cuh"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/adaptive_max_pool1d_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateAdaptiveMaxPool1dDescriptor(
    infiniopHandle_t handle,
    infiniopAdaptiveMaxPool1dDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t output_size) {

#define CREATE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                             \
        return op::adaptive_max_pool1d::NAMESPACE::Descriptor::create(                     \
            handle,                                                                        \
            reinterpret_cast<op::adaptive_max_pool1d::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                                        \
            x_desc,                                                                        \
            output_size)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__INFINI_C infiniStatus_t infiniopGetAdaptiveMaxPool1dWorkspaceSize(
    infiniopAdaptiveMaxPool1dDescriptor_t desc,
    size_t *size) {
#define GET(CASE, NAMESPACE)                                                                               \
    case CASE:                                                                                             \
        *size = reinterpret_cast<op::adaptive_max_pool1d::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__INFINI_C infiniStatus_t infiniopAdaptiveMaxPool1d(
    infiniopAdaptiveMaxPool1dDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {
#define CALCULATE(CASE, NAMESPACE)                                                                  \
    case CASE:                                                                                      \
        return reinterpret_cast<op::adaptive_max_pool1d::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__INFINI_C infiniStatus_t infiniopDestroyAdaptiveMaxPool1dDescriptor(
    infiniopAdaptiveMaxPool1dDescriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                         \
    case CASE:                                                                           \
        delete reinterpret_cast<op::adaptive_max_pool1d::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DESTROY(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
    }
#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
