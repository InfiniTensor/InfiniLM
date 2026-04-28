#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/smooth_l1_loss.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/smooth_l1_loss_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/smooth_l1_loss_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/smooth_l1_loss_metax.h"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/smooth_l1_loss_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
__INFINI_C infiniStatus_t infiniopCreateSmoothL1LossDescriptor(
    infiniopHandle_t handle,
    infiniopSmoothL1LossDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t output,
    infiniopTensorDescriptor_t input,
    infiniopTensorDescriptor_t target,
    float beta,
    int reduction) {

#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::smooth_l1_loss::NAMESPACE::Descriptor::create(                     \
            handle,                                                                   \
            reinterpret_cast<op::smooth_l1_loss::NAMESPACE::Descriptor **>(desc_ptr), \
            output,                                                                   \
            input,                                                                    \
            target,                                                                   \
            beta,                                                                     \
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
__INFINI_C infiniStatus_t infiniopGetSmoothL1LossWorkspaceSize(infiniopSmoothL1LossDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::smooth_l1_loss::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
__INFINI_C infiniStatus_t infiniopSmoothL1Loss(
    infiniopSmoothL1LossDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *target,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                           \
        return reinterpret_cast<const op::smooth_l1_loss::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, output, input, target, stream)

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
__INFINI_C infiniStatus_t infiniopDestroySmoothL1LossDescriptor(infiniopSmoothL1LossDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                           \
    case CASE:                                                                            \
        delete reinterpret_cast<const op::smooth_l1_loss::NAMESPACE::Descriptor *>(desc); \
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
