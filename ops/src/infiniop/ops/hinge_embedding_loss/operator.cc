#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/hinge_embedding_loss.h"

#ifdef ENABLE_CPU_API
#include "cpu/hinge_embedding_loss_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/hinge_embedding_loss_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/hinge_embedding_loss_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/hinge_embedding_loss_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateHingeEmbeddingLossDescriptor(
    infiniopHandle_t handle,
    infiniopHingeEmbeddingLossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t target_desc,
    double margin,
    int reduction) {

#define CREATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                              \
        return op::hinge_embedding_loss::NAMESPACE::Descriptor::create(                     \
            handle,                                                                         \
            reinterpret_cast<op::hinge_embedding_loss::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                                         \
            input_desc,                                                                     \
            target_desc,                                                                    \
            margin,                                                                         \
            reduction)

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
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetHingeEmbeddingLossWorkspaceSize(infiniopHingeEmbeddingLossDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                                \
    case CASE:                                                                                              \
        *size = reinterpret_cast<op::hinge_embedding_loss::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia)
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax)
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__INFINI_C infiniStatus_t infiniopHingeEmbeddingLoss(
    infiniopHingeEmbeddingLossDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *input,
    const void *target,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                 \
        return reinterpret_cast<const op::hinge_embedding_loss::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, input, target, stream)

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
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t
infiniopDestroyHingeEmbeddingLossDescriptor(infiniopHingeEmbeddingLossDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                  \
        delete reinterpret_cast<const op::hinge_embedding_loss::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
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
