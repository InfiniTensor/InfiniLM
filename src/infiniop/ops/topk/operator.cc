#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/topk.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "cuda/topk.cuh"
#endif

// ... (Add other backends like cpu, ascend, etc. here if they are implemented)

__C infiniStatus_t
infiniopCreateTopKDescriptor(infiniopHandle_t handle,
                             infiniopTopKDescriptor_t *desc_ptr,
                             infiniopTensorDescriptor_t input_desc,
                             infiniopTensorDescriptor_t output_val_desc,
                             infiniopTensorDescriptor_t output_ind_desc, int k) {

#define CREATE(CASE, NAMESPACE)                                                \\
    case CASE:                                                                 \\
        return op::topk::NAMESPACE::Descriptor::create(                        \\
            handle,                                                            \\
            reinterpret_cast<op::topk::NAMESPACE::Descriptor **>(desc_ptr),     \\
            input_desc, output_val_desc, output_ind_desc, k)

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
infiniopDestroyTopKDescriptor(infiniopTopKDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                \\
    case CASE:                                                                 \\
        delete reinterpret_cast<const op::topk::NAMESPACE::Descriptor *>(      \\
            desc);                                                             \\
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

__C infiniStatus_t infiniopGetTopKWorkspaceSize(infiniopTopKDescriptor_t desc,
                                                  size_t *size) {
#define GET_WORKSPACE_SIZE(CASE, NAMESPACE)                                    \\
    case CASE:                                                                 \\
        *size = reinterpret_cast<const op::topk::NAMESPACE::Descriptor *>(     \\
                    desc)                                                      \\
                    ->getWorkspaceSize();                                      \\
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
        GET_WORKSPACE_SIZE(INFINI_DEVICE_NVIDIA, cuda);
#endif
    // ... (Add cases for other devices)
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET_WORKSPACE_SIZE
}

__C infiniStatus_t
infiniopTopK(infiniopTopKDescriptor_t desc, void *workspace,
             size_t workspace_size, void *output_val, void *output_ind,
             const void *input, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \\
    case CASE:                                                                 \\
        return reinterpret_cast<const op::topk::NAMESPACE::Descriptor *>(      \\
                   desc)                                                       \\
            ->calculate(input, output_val, output_ind, workspace, stream)

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