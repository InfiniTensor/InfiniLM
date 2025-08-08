#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/add.h"

#ifdef ENABLE_CPU_API
#include "cpu/add_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/add_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/add_metax.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/add_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateAddDescriptor(
    infiniopHandle_t handle,
    infiniopAddDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                            \
    case CASE:                                                             \
        return op::add::NAMESPACE::Descriptor::create(                     \
            handle,                                                        \
            reinterpret_cast<op::add::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                        \
            {a_desc,                                                       \
             b_desc})

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
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetAddWorkspaceSize(infiniopAddDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                               \
    case CASE:                                                                             \
        *size = reinterpret_cast<op::add::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

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
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopAdd(
    infiniopAddDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                            \
    case CASE:                                                                \
        return reinterpret_cast<const op::add::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, c, {a, b}, stream)

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
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyAddDescriptor(infiniopAddDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::add::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_KUNLUN_API
        DELETE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
