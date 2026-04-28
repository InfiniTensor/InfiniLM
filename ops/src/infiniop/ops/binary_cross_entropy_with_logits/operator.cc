#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/binary_cross_entropy_with_logits.h"

// 引入各硬件后端的 Descriptor 定义
#ifdef ENABLE_CPU_API
#include "cpu/binary_cross_entropy_with_logits_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API)
#include "nvidia/binary_cross_entropy_with_logits_nvidia.cuh"
#endif
#ifdef ENABLE_METAX_API
#include "metax/binary_cross_entropy_with_logits_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/binary_cross_entropy_with_logits_moore.h"
#endif

// -----------------------------------------------------------------------------
// 1. 创建描述符
// -----------------------------------------------------------------------------
__INFINI_C infiniStatus_t infiniopCreateBCEWithLogitsDescriptor(
    infiniopHandle_t handle,
    infiniopBCEWithLogitsDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t logits_desc,
    infiniopTensorDescriptor_t target_desc,
    infiniopTensorDescriptor_t weight_desc,
    infiniopTensorDescriptor_t pos_weight_desc,
    infiniopReduction_t reduction) {

#define CREATE(CASE, NAMESPACE)                                                                                                              \
    case CASE:                                                                                                                               \
        return op::bce_with_logits::NAMESPACE::Descriptor::create(handle,                                                                    \
                                                                  reinterpret_cast<op::bce_with_logits::NAMESPACE::Descriptor **>(desc_ptr), \
                                                                  out_desc, logits_desc, target_desc, weight_desc, pos_weight_desc, reduction)

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
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
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
// 2. 获取 Workspace 大小
// -----------------------------------------------------------------------------
__INFINI_C infiniStatus_t infiniopGetBCEWithLogitsWorkspaceSize(
    infiniopBCEWithLogitsDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                                 \
    case CASE:                                                                                               \
        *size = reinterpret_cast<const op::bce_with_logits::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
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
}

// -----------------------------------------------------------------------------
// 3. 执行计算
// -----------------------------------------------------------------------------
__INFINI_C infiniStatus_t infiniopBCEWithLogits(
    infiniopBCEWithLogitsDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *out,
    const void *logits,
    const void *target,
    const void *weight,
    const void *pos_weight,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                            \
        return reinterpret_cast<const op::bce_with_logits::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, out, logits, target, weight, pos_weight, stream)

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
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
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
// 4. 销毁描述符
// -----------------------------------------------------------------------------
__INFINI_C infiniStatus_t infiniopDestroyBCEWithLogitsDescriptor(infiniopBCEWithLogitsDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                            \
    case CASE:                                                                             \
        delete reinterpret_cast<const op::bce_with_logits::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

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
#ifdef ENABLE_HYGON_API
        DELETE(INFINI_DEVICE_HYGON, nvidia);
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
