#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_combine.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/moe_combine.cuh"
#endif

// ... (Add other backends like cpu, ascend, etc. here if they are implemented)

__C infiniStatus_t infiniopCreateMoECombineDescriptor(
    infiniopHandle_t handle, infiniopMoECombineDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t permuted_input_desc,
    infiniopTensorDescriptor_t gating_weights_desc,
    infiniopTensorDescriptor_t aux_info_desc,
    infiniopTensorDescriptor_t output_desc) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                    \
        return op::moe_combine::NAMESPACE::Descriptor::create(                    \
            handle,                                                               \
            reinterpret_cast<op::moe_combine::NAMESPACE::Descriptor **>(desc_ptr), \
            permuted_input_desc, gating_weights_desc, aux_info_desc, output_desc)

    switch (handle->device) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    // ... (Add cases for other devices)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t
infiniopDestroyMoECombineDescriptor(infiniopMoECombineDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                          \
    case CASE:                                                                           \
        delete reinterpret_cast<const op::moe_combine::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    // ... (Add cases for other devices)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}

__C infiniStatus_t infiniopMoECombine(
    infiniopMoECombineDescriptor_t desc, 
    const void *permuted_input, const void *gating_weights,
    const void *aux_info,void *output, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                               \
    case CASE:                                                                    \
        return reinterpret_cast<const op::moe_combine::NAMESPACE::Descriptor *>( \
                   desc)                                                          \
            ->calculate(permuted_input, gating_weights, aux_info, output, stream)

    switch (desc->device_type) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    // ... (Add cases for other devices)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
} 