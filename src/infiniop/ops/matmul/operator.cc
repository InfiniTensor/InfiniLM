#include "infiniop/ops/matmul.h"

#ifdef ENABLE_CPU_API
#include "cpu/matmul_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/matmul_cuda.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/matmul_bang.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/matmul_ascend.h"
#endif

__C infiniopStatus_t infiniopCreateMatmulDescriptor(
    infiniopHandle_t handle,
    infiniopMatmulDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                           \
    case CASE:                                                            \
        return matmul::NAMESPACE::Descriptor::create(                     \
            handle,                                                       \
            reinterpret_cast<matmul::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                       \
            a_desc,                                                       \
            b_desc)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniopStatus_t
infiniopGetMatmulWorkspaceSize(
    infiniopMatmulDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                   \
    case CASE:                                                                                 \
        *size = reinterpret_cast<matmul::NAMESPACE::Descriptor const *>(desc)->workspace_size; \
        return INFINIOP_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        GET(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniopStatus_t infiniopMatmul(
    infiniopMatmulDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *c,
    void const *a,
    void const *b,
    float alpha,
    float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                           \
    case CASE:                                                               \
        return reinterpret_cast<matmul::NAMESPACE::Descriptor const *>(desc) \
            ->calculate(workspace, workspace_size,                           \
                        c, beta,                                             \
                        a, b, alpha,                                         \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniopStatus_t
infiniopDestroyMatmulDescriptor(infiniopMatmulDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                               \
    case CASE:                                                                \
        delete reinterpret_cast<matmul::NAMESPACE::Descriptor const *>(desc); \
        return INFINIOP_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend);
#endif

    default:
        return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
