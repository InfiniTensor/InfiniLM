#include "../../operator.h"
#include "../../handle.h"
#include "infinicore.h"
#include "infiniop/ops/topk.h"
#include "topk.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/topk.cuh"
#endif

// ... (Add other backends like cpu, ascend, etc. here if they are implemented)

__C infiniStatus_t
infiniopCreateTopKDescriptor(infiniopHandle_t handle,
                             infiniopTopKDescriptor_t *desc_ptr,
                             infiniopTensorDescriptor_t input_desc,
                             infiniopTensorDescriptor_t output_val_desc,
                             infiniopTensorDescriptor_t output_ind_desc,
                             infiniopTensorDescriptor_t bias_desc, int k,
                             int strategy, int n_group,
                             int topk_group) {

#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::topk::NAMESPACE::Descriptor::create(                        \
            handle,                                                            \
            reinterpret_cast<op::topk::NAMESPACE::Descriptor **>(desc_ptr),     \
            input_desc, output_val_desc, output_ind_desc, bias_desc, k,         \
            strategy, n_group, topk_group)

    switch (handle->device) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
    CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_BANG_API
    CREATE(INFINI_DEVICE_BANG, bang);
#endif
#ifdef ENABLE_XPU_API
    CREATE(INFINI_DEVICE_XPU, xpu);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__C infiniStatus_t infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc) {
    if (desc == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    delete desc;
    return INFINI_STATUS_SUCCESS;
}

__C size_t infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc) {
    if (desc == nullptr) {
        return 0;
    }
    // Cast to the appropriate descriptor type to call the method
    switch (desc->device_type) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
    case INFINI_DEVICE_NVIDIA:
        return reinterpret_cast<op::topk::nvidia::Descriptor *>(desc)
            ->getWorkspaceSize();
#endif
    default:
        return 0;
    }
}

__C infiniStatus_t infiniopTopK(infiniopTopKDescriptor_t desc,
                                const void *input, void *output_val,
                                void *output_ind, const void *bias,
                                void *workspace, void *stream) {
    if (desc == nullptr) {
        return INFINI_STATUS_BAD_PARAM;
    }
    // Cast to the appropriate descriptor type to call the method
    switch (desc->device_type) {
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
    case INFINI_DEVICE_NVIDIA:
        return reinterpret_cast<op::topk::nvidia::Descriptor *>(desc)
            ->calculate(input, output_val, output_ind, bias, workspace, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
} 