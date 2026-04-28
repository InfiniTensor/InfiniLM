#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/silu_and_mul.h"

#ifdef ENABLE_MOORE_API
#include "moore/silu_and_mul_moore.h"
#endif

__INFINI_C infiniStatus_t infiniopCreateSiluAndMulDescriptor(
    infiniopHandle_t handle,
    infiniopSiluAndMulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {

#define CREATE(CASE, NAMESPACE)                                                     \
    case CASE:                                                                      \
        return op::silu_and_mul::NAMESPACE::Descriptor::create(                     \
            handle,                                                                 \
            reinterpret_cast<op::silu_and_mul::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                                 \
            x_desc);

    switch (handle->device) {
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__INFINI_C infiniStatus_t infiniopGetSiluAndMulWorkspaceSize(infiniopSiluAndMulDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                              \
    case CASE:                                                                                            \
        *size = reinterpret_cast<const op::silu_and_mul::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__INFINI_C infiniStatus_t infiniopSiluAndMul(
    infiniopSiluAndMulDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *y,
    const void *x,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                                 \
    case CASE:                                                                                     \
        return reinterpret_cast<const op::silu_and_mul::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, stream);

    switch (desc->device_type) {
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__INFINI_C infiniStatus_t infiniopDestroySiluAndMulDescriptor(infiniopSiluAndMulDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                        \
    case CASE:                                                                          \
        delete reinterpret_cast<const op::silu_and_mul::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, moore);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DESTROY
}
