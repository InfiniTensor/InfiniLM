#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/kthvalue.h"

// --- 后端实现头文件 ---
#ifdef ENABLE_CPU_API
#include "cpu/kthvalue_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API)
#include "nvidia/kthvalue_nvidia.cuh"
#endif

#ifdef ENABLE_METAX_API
#include "metax/kthvalue_metax.h"
#endif

#ifdef ENABLE_MOORE_API
#include "moore/kthvalue_moore.h"
#endif

extern "C" {

// =======================================================================
// 1. 创建算子描述符
// =======================================================================
__INFINI_C infiniStatus_t infiniopCreateKthvalueDescriptor(
    infiniopHandle_t handle,
    infiniopKthvalueDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t values,
    infiniopTensorDescriptor_t indices,
    infiniopTensorDescriptor_t input,
    int k,
    int dim,
    int keepdim) {

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::kthvalue::NAMESPACE::Descriptor::create(                     \
            handle,                                                             \
            reinterpret_cast<op::kthvalue::NAMESPACE::Descriptor **>(desc_ptr), \
            values,                                                             \
            indices,                                                            \
            input,                                                              \
            k,                                                                  \
            dim,                                                                \
            keepdim)

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
__INFINI_C infiniStatus_t infiniopGetKthvalueWorkspaceSize(infiniopKthvalueDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                  \
        *size = reinterpret_cast<op::kthvalue::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
__INFINI_C infiniStatus_t infiniopKthvalue(
    infiniopKthvalueDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *values,
    void *indices,
    const void *input,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                     \
        return reinterpret_cast<const op::kthvalue::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, values, indices, input, stream)

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
__INFINI_C infiniStatus_t infiniopDestroyKthvalueDescriptor(infiniopKthvalueDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        delete reinterpret_cast<const op::kthvalue::NAMESPACE::Descriptor *>(desc); \
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
