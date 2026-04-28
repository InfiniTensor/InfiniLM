#include "../../../operator.h"
#include "../../../handle.h"
#include "infiniop/ops/quant/per_channel_quant_int8.h"

#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_QY_API)
#include "nvidia/per_channel_quant_int8_nvidia.cuh"
#endif
#if defined(ENABLE_MOORE_API)
#include "moore/per_channel_quant_int8_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreatePerChannelQuantI8Descriptor(infiniopHandle_t handle,
                                                                    infiniopPerChannelQuantI8Descriptor_t *desc_ptr,
                                                                    infiniopTensorDescriptor_t x_packed_desc,
                                                                    infiniopTensorDescriptor_t x_scale_desc,
                                                                    infiniopTensorDescriptor_t x_zero_desc,
                                                                    infiniopTensorDescriptor_t x_desc) {
#define CREATE(CASE, NAMESPACE)                                                               \
    case CASE:                                                                                \
        return op::per_channel_quant_int8::NAMESPACE::Descriptor::create(                     \
            handle,                                                                           \
            reinterpret_cast<op::per_channel_quant_int8::NAMESPACE::Descriptor **>(desc_ptr), \
            x_packed_desc,                                                                    \
            x_scale_desc,                                                                     \
            x_zero_desc,                                                                      \
            x_desc);
    switch (handle->device) {
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        CREATE(INFINI_DEVICE_QY, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetPerChannelQuantI8WorkspaceSize(infiniopPerChannelQuantI8Descriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#define GET(CASE, NAMESPACE)                                                                                     \
    case CASE:                                                                                                   \
        *size = reinterpret_cast<op::per_channel_quant_int8::NAMESPACE::Descriptor *>(desc)->minWorkspaceSize(); \
        return INFINI_STATUS_SUCCESS;
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        GET(INFINI_DEVICE_QY, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef GET
}

__INFINI_C infiniStatus_t infiniopPerChannelQuantI8(infiniopPerChannelQuantI8Descriptor_t desc,
                                                    void *workspace,
                                                    size_t workspace_size,
                                                    void *x_packed,
                                                    void *x_scale,
                                                    void *x_zero,
                                                    const void *x,
                                                    void *stream) {
#define QUANT(CASE, NAMESPACE)                                                                         \
    case CASE:                                                                                         \
        return reinterpret_cast<op::per_channel_quant_int8::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, x_packed, x_scale, x_zero, x, stream);

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        QUANT(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        QUANT(INFINI_DEVICE_QY, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        QUANT(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef QUANT
}

__INFINI_C infiniStatus_t infiniopDestroyPerChannelQuantI8Descriptor(infiniopPerChannelQuantI8Descriptor_t desc) {
#define DESTROY(CASE, NAMESPACE)                                                            \
    case CASE:                                                                              \
        delete reinterpret_cast<op::per_channel_quant_int8::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_NVIDIA_API
        DESTROY(INFINI_DEVICE_NVIDIA, nvidia)
#endif
#ifdef ENABLE_QY_API
        DESTROY(INFINI_DEVICE_QY, nvidia)
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore)
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
#undef DESTROY
}
