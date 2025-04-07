#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/gemm.h"

#ifdef ENABLE_CPU_API
#include "cpu/gemm_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/gemm_cuda.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/gemm_bang.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/gemm_ascend.h"
#endif
#ifdef ENABLE_METAX_API
#include "maca/gemm_maca.h"
#endif
#ifdef ENABLE_MOORE_API
#include "musa/gemm_musa.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/gemm_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateGemmDescriptor(
    infiniopHandle_t handle,
    infiniopGemmDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::gemm::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::gemm::NAMESPACE::Descriptor **>(desc_ptr), \
            c_desc,                                                         \
            a_desc,                                                         \
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
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, maca);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, musa);
#endif

#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t
infiniopGetGemmWorkspaceSize(
    infiniopGemmDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<const op::gemm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

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
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, maca);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopGemm(
    infiniopGemmDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    float alpha,
    float beta,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::gemm::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                             \
                        c, beta,                                               \
                        a, b, alpha,                                           \
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
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, maca);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyGemmDescriptor(infiniopGemmDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::gemm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

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
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, maca);
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        DELETE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
