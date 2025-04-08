#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rms_norm.h"

#ifdef ENABLE_CPU_API
#include "cpu/rms_norm_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/rms_norm_cuda.cuh"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/rms_norm_aclnn.h"
#endif
#ifdef ENABLE_METAX_API
#include "maca/rms_norm_maca.cuh"
#endif
#ifdef ENABLE_MOORE_API
#include "musa/rms_norm_musa.cuh"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/rms_norm_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateRMSNormDescriptor(
    infiniopHandle_t handle,
    infiniopRMSNormDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {

#define CREATE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        return op::rms_norm::NAMESPACE::Descriptor::create(                     \
            handle,                                                             \
            reinterpret_cast<op::rms_norm::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc,                                                             \
            x_desc,                                                             \
            w_desc,                                                             \
            epsilon);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda)
#endif
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun)
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCreateRMSNormDescriptor((BangHandle_t)handle, (RMSNormBangDescriptor_t *)desc_ptr, y_desc, x_desc, w_desc, epsilon);
    }
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend)
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, maca)
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, musa)
#endif
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetRMSNormWorkspaceSize(infiniopRMSNormDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                    \
    case CASE:                                                                                  \
        *size = reinterpret_cast<op::rms_norm::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        GET(INFINI_DEVICE_NVIDIA, cuda)
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun)
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangGetRMSNormWorkspaceSize((RMSNormBangDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend)
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, maca)
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, musa)
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRMSNorm(infiniopRMSNormDescriptor_t desc, void *workspace, size_t workspace_size,
                                   void *y, const void *x, const void *w, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                           \
        return reinterpret_cast<op::rms_norm::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, y, x, w, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda)
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun)
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangRMSNorm((RMSNormBangDescriptor_t)desc, workspace, workspace_size, y, x, w, stream);
    }
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend)
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, maca)
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, musa)
#endif
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyRMSNormDescriptor(infiniopRMSNormDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                              \
    case CASE:                                                                \
        delete reinterpret_cast<op::rms_norm::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_CUDA_API
        DESTROY(INFINI_DEVICE_NVIDIA, cuda)
#endif
#ifdef ENABLE_KUNLUN_API
        DESTROY(INFINI_DEVICE_KUNLUN, kunlun)
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangDestroyRMSNormDescriptor((RMSNormBangDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_ASCEND_API
        DESTROY(INFINI_DEVICE_ASCEND, ascend)
#endif
#ifdef ENABLE_METAX_API
        DESTROY(INFINI_DEVICE_METAX, maca)
#endif
#ifdef ENABLE_MOORE_API
        DESTROY(INFINI_DEVICE_MOORE, musa)
#endif
    }

#undef DESTROY

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
