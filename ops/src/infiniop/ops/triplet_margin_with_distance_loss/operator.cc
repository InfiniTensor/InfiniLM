#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/triplet_margin_with_distance_loss.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/triplet_margin_with_distance_loss_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/triplet_margin_with_distance_loss_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/triplet_margin_with_distance_loss_metax.h"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/triplet_margin_with_distance_loss_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
__INFINI_C infiniStatus_t infiniopCreateTripletMarginWithDistanceLossDescriptor(
    infiniopHandle_t handle,
    infiniopTripletMarginWithDistanceLossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t anchor,
    infiniopTensorDescriptor_t positive,
    infiniopTensorDescriptor_t negative,
    float margin,
    int swap,
    int reduction) {

#define CREATE(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                           \
        return op::triplet_margin_with_distance_loss::NAMESPACE::Descriptor::create(                     \
            handle,                                                                                      \
            reinterpret_cast<op::triplet_margin_with_distance_loss::NAMESPACE::Descriptor **>(desc_ptr), \
            output,                                                                                      \
            anchor,                                                                                      \
            positive,                                                                                    \
            negative,                                                                                    \
            margin,                                                                                      \
            swap,                                                                                        \
            reduction)

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

// =======================================================================
// 2. 获取 Workspace 大小
// =======================================================================
__INFINI_C infiniStatus_t infiniopGetTripletMarginWithDistanceLossWorkspaceSize(
    infiniopTripletMarginWithDistanceLossDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                                             \
    case CASE:                                                                                                           \
        *size = reinterpret_cast<op::triplet_margin_with_distance_loss::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
}

// =======================================================================
// 3. 执行计算 (Calculate)
// =======================================================================
__INFINI_C infiniStatus_t infiniopTripletMarginWithDistanceLoss(
    infiniopTripletMarginWithDistanceLossDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *anchor,
    const void *positive,
    const void *negative,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                              \
        return reinterpret_cast<const op::triplet_margin_with_distance_loss::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, anchor, positive, negative, stream)

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

// =======================================================================
// 4. 销毁描述符
// =======================================================================
__INFINI_C infiniStatus_t infiniopDestroyTripletMarginWithDistanceLossDescriptor(
    infiniopTripletMarginWithDistanceLossDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                                              \
    case CASE:                                                                                               \
        delete reinterpret_cast<const op::triplet_margin_with_distance_loss::NAMESPACE::Descriptor *>(desc); \
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

} // extern "C"
