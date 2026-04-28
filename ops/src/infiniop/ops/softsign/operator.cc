#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/softsign.h" // 必须包含上面定义的头文件

#ifdef ENABLE_CPU_API
#include "cpu/softsign_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/softsign_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/softsign_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/softsign_moore.h"
#endif

// -----------------------------------------------------------------------------
// Struct Definition
// -----------------------------------------------------------------------------

// 修正：使用 infiniDevice_t
struct InfiniopSoftsignDescriptor {
    infiniDevice_t device_type;
};

// -----------------------------------------------------------------------------
// Create Descriptor
// -----------------------------------------------------------------------------

__INFINI_C infiniStatus_t infiniopCreateSoftsignDescriptor(
    infiniopHandle_t handle,
    infiniopSoftsignDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

// 使用 {x_desc} 构建 vector
#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE: {                                                                \
        auto status = op::softsign::NAMESPACE::Descriptor::create(              \
            handle,                                                             \
            reinterpret_cast<op::softsign::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                             \
            {x_desc});                                                          \
        if (status == INFINI_STATUS_SUCCESS) {                                  \
            (*desc_ptr)->device_type = CASE;                                    \
        }                                                                       \
        return status;                                                          \
    }

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
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

// -----------------------------------------------------------------------------
// Get Workspace Size
// -----------------------------------------------------------------------------

__INFINI_C infiniStatus_t infiniopGetSoftsignWorkspaceSize(infiniopSoftsignDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                  \
        *size = reinterpret_cast<op::softsign::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

// -----------------------------------------------------------------------------
// Execute (Calculate)
// -----------------------------------------------------------------------------

__INFINI_C infiniStatus_t infiniopSoftsign(
    infiniopSoftsignDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {

// 使用 {x} 构建 vector
#define CALCULATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                     \
        return reinterpret_cast<const op::softsign::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, {x}, stream)

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
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

// -----------------------------------------------------------------------------
// Destroy Descriptor
// -----------------------------------------------------------------------------

__INFINI_C infiniStatus_t
infiniopDestroySoftsignDescriptor(infiniopSoftsignDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        delete reinterpret_cast<const op::softsign::NAMESPACE::Descriptor *>(desc); \
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
#ifdef ENABLE_QY_API
        DELETE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
