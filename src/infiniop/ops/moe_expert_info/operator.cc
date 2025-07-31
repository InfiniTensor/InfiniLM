#include "../../handle.h"
#include "../../operator.h"
#include "infiniop/ops/moe_expert_info.h"

// Include backend-specific headers, guarded by compiler flags
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/moe_expert_info.cuh" // Assuming this is the private C++ header for the nvidia backend
#endif

// Add other backends like cpu, ascend, etc. here if they are implemented

__C infiniStatus_t infiniopCreateMoEExpertInfoDescriptor(
    infiniopHandle_t handle, infiniopMoEExpertInfoDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_ind_desc,
    infiniopTensorDescriptor_t expert_counts_desc,
    infiniopTensorDescriptor_t expert_offsets_desc) {

// Macro to call the static create method of the backend-specific Descriptor
#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::moe_expert_info::NAMESPACE::Descriptor::create(              \
            handle,                                                            \
            reinterpret_cast<op::moe_expert_info::NAMESPACE::Descriptor **>(    \
                desc_ptr),                                                     \
            topk_ind_desc, expert_counts_desc, expert_offsets_desc)

    switch (handle->device) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    // Add cases for other devices (e.g., CREATE(INFINI_DEVICE_ASCEND, ascend))
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopDestroyMoEExpertInfoDescriptor(
    infiniopMoEExpertInfoDescriptor_t desc) {

// Macro to delete the backend-specific Descriptor object
#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::moe_expert_info::NAMESPACE::Descriptor *>( \
            desc);                                                             \
        return INFINI_STATUS_SUCCESS

    // The descriptor must not be null
    if (!desc) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    switch (desc->device_type) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    // Add cases for other devices
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}

__C infiniStatus_t infiniopMoEExpertInfoCalculate(
    infiniopHandle_t handle, infiniopMoEExpertInfoDescriptor_t desc,
    const void *topk_ind, void *expert_counts, void *expert_offsets,
    infinirtStream_t stream) {

// Macro to call the calculate method of the backend-specific Descriptor
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::moe_expert_info::NAMESPACE::Descriptor *>( \
                   desc)                                                       \
            ->calculate(topk_ind, expert_counts, expert_offsets, stream)

    // The descriptor must not be null
    if (!desc) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    switch (desc->device_type) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    // Add cases for other devices
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
}