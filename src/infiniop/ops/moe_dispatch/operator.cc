#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/moe_dispatch.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "cuda/moe_dispatch.cuh"
#endif

// ... (Add other backends like cpu, ascend, etc. here if they are implemented)

__C infiniStatus_t infiniopCreateMoEDispatchDescriptor(
    infiniopHandle_t handle, infiniopMoEDispatchDescriptor_t *desc_ptr,
    int num_experts, infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t indices_desc,
    infiniopTensorDescriptor_t permuted_output_desc,
    infiniopTensorDescriptor_t aux_info_desc) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                    \
        return op::moe_dispatch::NAMESPACE::Descriptor::create(                    \
            handle,                                                               \
            reinterpret_cast<op::moe_dispatch::NAMESPACE::Descriptor **>(         \
                desc_ptr),                                                        \
            num_experts, input_desc, indices_desc, permuted_output_desc,          \
            aux_info_desc)

    switch (handle->device) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
    // ... (Add cases for other devices)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t
infiniopDestroyMoEDispatchDescriptor(infiniopMoEDispatchDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        delete reinterpret_cast<const op::moe_dispatch::NAMESPACE::Descriptor *>( \
            desc);                                                                  \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif
    // ... (Add cases for other devices)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DELETE
}

__C infiniStatus_t infiniopMoEDispatch(
    infiniopMoEDispatchDescriptor_t desc, void *permuted_output,
    void *aux_info, const void *input, const void *indices,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                      \
        return reinterpret_cast<const op::moe_dispatch::NAMESPACE::Descriptor *>( \
                   desc)                                                            \
            ->calculate(input, indices, permuted_output, aux_info, stream)

    switch (desc->device_type) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
    // ... (Add cases for other devices)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CALCULATE
} 