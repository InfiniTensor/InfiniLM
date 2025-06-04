#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/rope.h"

#ifdef ENABLE_CPU_API
#include "cpu/rope_cpu.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/rope_cuda.cuh"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/rope_ascend.h"
#endif

__C infiniStatus_t infiniopCreateRoPEDescriptor(
    infiniopHandle_t handle,
    infiniopRoPEDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y,
    infiniopTensorDescriptor_t x,
    infiniopTensorDescriptor_t pos_ids,
    infiniopTensorDescriptor_t sin_table,
    infiniopTensorDescriptor_t cos_table) {

#define CREATE(CASE, NAMESPACE)                                             \
    case CASE:                                                              \
        return op::rope::NAMESPACE::Descriptor::create(                     \
            handle,                                                         \
            reinterpret_cast<op::rope::NAMESPACE::Descriptor **>(desc_ptr), \
            y,                                                              \
            x,                                                              \
            pos_ids,                                                        \
            sin_table,                                                      \
            cos_table)

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCreateRoPEDescriptor((BangHandle_t)handle,
                                        (RoPEBangDescriptor_t *)desc_ptr, t,
                                        pos_ids, sin_table, cos_table);
    }
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaCreateRoPEDescriptor((MacaHandle_t)handle,
                                        (RoPEMacaDescriptor_t *)desc_ptr, t,
                                        pos_ids, sin_table, cos_table);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaCreateRoPEDescriptor((MusaHandle_t)handle,
                                        (RoPEMusaDescriptor_t *)desc_ptr, t,
                                        pos_ids, sin_table, cos_table);
    }
#endif
    }

#undef CREATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetRoPEWorkspaceSize(infiniopRoPEDescriptor_t desc,
                                                size_t *size) {
#define GET(CASE, NAMESPACE)                                                                      \
    case CASE:                                                                                    \
        *size = reinterpret_cast<const op::rope::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        GET(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangGetRoPEWorkspaceSize((RoPEBangDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaGetRoPEWorkspaceSize((RoPEMacaDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaGetRoPEWorkspaceSize((RoPEMusaDescriptor_t)desc, size);
    }
#endif
    }

#undef GET

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRoPE(
    infiniopRoPEDescriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *y,
    const void *x,
    const void *pos_ids,
    const void *sin_table,
    const void *cos_table,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                             \
    case CASE:                                                                 \
        return reinterpret_cast<const op::rope::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size, y, x, pos_ids, sin_table, cos_table, stream)

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangRoPE((RoPEBangDescriptor_t)desc, workspace, workspace_size,
                        t, pos_ids, sin_table, cos_table, stream);
    }
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaRoPE((RoPEMacaDescriptor_t)desc, workspace, workspace_size,
                        t, pos_ids, sin_table, cos_table, stream);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaRoPE((RoPEMusaDescriptor_t)desc, workspace, workspace_size,
                        t, pos_ids, sin_table, cos_table, stream);
    }
#endif
    }

#undef CALCULATE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t
infiniopDestroyRoPEDescriptor(infiniopRoPEDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                 \
    case CASE:                                                                  \
        delete reinterpret_cast<const op::rope::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangDestroyRoPEDescriptor((RoPEBangDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaDestroyRoPEDescriptor((RoPEMacaDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaDestroyRoPEDescriptor((RoPEMusaDescriptor_t)desc);
    }
#endif
    }

#undef DELETE

    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
