#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/add_rms_norm.h"

#ifdef ENABLE_CPU_API
#include "cpu/add_rms_norm_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API) || defined(ENABLE_QY_API) || defined(ENABLE_HYGON_API) || defined(ENABLE_ALI_API)
#include "nvidia/add_rms_norm_nvidia.cuh"
#endif
#ifdef ENABLE_ASCEND_API
// TODO: Add Ascend implementation
// #include "ascend/add_rms_norm_aclnn.h"
#endif
#ifdef ENABLE_CAMBRICON_API
// TODO: Add Cambricon implementation
// #include "bang/add_rms_norm_bang.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/add_rms_norm_metax.cuh"
#endif
#ifdef ENABLE_MOORE_API
#include "moore/add_rms_norm_moore.h"
#endif
#ifdef ENABLE_KUNLUN_API
// TODO: Add Kunlun implementation
// #include "kunlun/add_rms_norm_kunlun.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateAddRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopAddRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t residual_out_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t weight_desc,
    float epsilon) {

#define CREATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        return op::add_rms_norm::NAMESPACE::Descriptor::create(                     \
            handle,                                                                 \
            reinterpret_cast<op::add_rms_norm::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                                 \
            residual_out_desc,                                                      \
            a_desc,                                                                 \
            b_desc,                                                                 \
            weight_desc,                                                            \
            epsilon)

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
#ifdef ENABLE_ALI_API
        CREATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CREATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        // CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetAddRMSNormWorkspaceSize(infiniopAddRMSNormDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                        \
    case CASE:                                                                                      \
        *size = reinterpret_cast<op::add_rms_norm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
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
#ifdef ENABLE_ALI_API
        GET(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        GET(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        // GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__INFINI_C infiniStatus_t infiniopAddRMSNorm(
    infiniopAddRMSNormDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    void *residual_out,
    const void *a,
    const void *b,
    const void *weight,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                         \
        return reinterpret_cast<const op::add_rms_norm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, residual_out, a, b, weight, stream)

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
#ifdef ENABLE_ALI_API
        CALCULATE(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_QY_API
        CALCULATE(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        CALCULATE(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        // CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroyAddRMSNormDescriptor(infiniopAddRMSNormDescriptor_t desc) {
    if (desc == nullptr) {
        return INFINI_STATUS_SUCCESS;
    }

#define DESTROY(CASE, NAMESPACE)                                                  \
    case CASE:                                                                    \
        delete reinterpret_cast<op::add_rms_norm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS

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
#ifdef ENABLE_ALI_API
        DESTROY(INFINI_DEVICE_ALI, nvidia);
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia);
#endif
#ifdef ENABLE_HYGON_API
        DESTROY(INFINI_DEVICE_HYGON, nvidia);
#endif
#ifdef ENABLE_KUNLUN_API
        // DESTROY(INFINI_DEVICE_KUNLUN, kunlun);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
