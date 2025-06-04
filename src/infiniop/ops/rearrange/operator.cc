#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rearrange.h"

#ifdef ENABLE_CPU_API
#include "cpu/rearrange_cpu.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/rearrange_ascend.h"
#endif

#ifdef ENABLE_CUDA_API
#include "cuda/rearrange_cuda.cuh"
#endif

__C infiniStatus_t infiniopCreateRearrangeDescriptor(
    infiniopHandle_t handle,
    infiniopRearrangeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src) {

#define CREATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                   \
        return op::rearrange::NAMESPACE::Descriptor::create(                     \
            handle,                                                              \
            reinterpret_cast<op::rearrange::NAMESPACE::Descriptor **>(desc_ptr), \
            dst,                                                                 \
            src)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif

#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopRearrange(
    infiniopRearrangeDescriptor_t desc,
    void *dst,
    const void *src,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                  \
    case CASE:                                                                      \
        return reinterpret_cast<const op::rearrange::NAMESPACE::Descriptor *>(desc) \
            ->calculate(dst, src, stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif

#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t infiniopDestroyRearrangeDescriptor(
    infiniopRearrangeDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                      \
    case CASE:                                                                       \
        delete reinterpret_cast<const op::rearrange::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend);
#endif

#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
