#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gqa.h" // Assumes the public C API header for GQA

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/gqa.cuh" // The backend-specific implementation
#endif

// Add other backend headers (e.g., cpu, ascend) here if they are implemented

/**
 * @brief C API function to create a GQA operator descriptor.
 *
 * This function acts as a dispatcher, calling the appropriate backend's
 * `create` method based on the device specified in the handle.
 */
__C infiniStatus_t infiniopCreateGQADescriptor(
    infiniopHandle_t handle, infiniopGQADescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t output_desc) {

// Macro to simplify creating backend-specific descriptors
#define CREATE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        return op::gqa::NAMESPACE::Descriptor::create(                         \
            handle,                                                            \
            reinterpret_cast<op::gqa::NAMESPACE::Descriptor **>(desc_ptr),       \
            q_desc, k_desc, v_desc, output_desc)

    switch (handle->device) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
    // Add cases for other devices (e.g., CREATE(INFINI_DEVICE_CPU, cpu);)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

/**
 * @brief C API function to destroy a GQA operator descriptor.
 *
 * Deallocates the memory for a previously created descriptor.
 */
__C infiniStatus_t
infiniopDestroyGQADescriptor(infiniopGQADescriptor_t desc) {

// Macro to simplify deleting backend-specific descriptors
#define DELETE(CASE, NAMESPACE)                                                \
    case CASE:                                                                 \
        delete reinterpret_cast<const op::gqa::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

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

/**
 * @brief C API function to execute the GQA operation.
 *
 * Calls the `calculate` method on the appropriate backend-specific descriptor.
 */
__C infiniStatus_t infiniopGQA(
    infiniopGQADescriptor_t desc,
    const void *q, const void *k, const void *v,
    void *output, void *stream) {

// Macro to simplify calling the backend-specific calculation
#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::gqa::NAMESPACE::Descriptor *>(       \
                   desc)                                                       \
            ->calculate(q, k, v, output, stream)

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
