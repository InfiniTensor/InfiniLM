#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/clip.h"

#ifdef ENABLE_CPU_API
#include "cpu/clip_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/clip_cuda.cuh"
#endif

__C infiniStatus_t infiniopCreateClipDescriptor(
    infiniopHandle_t handle,
    infiniopClipDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t min_val,
    infiniopTensorDescriptor_t max_val) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::clip::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::clip::NAMESPACE::Descriptor **>(desc_ptr), \
            y,                                                              \
            {x, min_val, max_val})

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopGetClipWorkspaceSize(infiniopClipDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                \
    case CASE:                                                                              \
        *size = reinterpret_cast<op::clip::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        GET(INFINI_DEVICE_NVIDIA, cuda)
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopClip(
    infiniopClipDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *min_val,
    const void *max_val,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::clip::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, {x, min_val, max_val}, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyClipDescriptor(infiniopClipDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::clip::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
