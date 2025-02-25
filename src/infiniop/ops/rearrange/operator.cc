#include "infiniop/ops/rearrange.h"

__C infiniStatus_t infiniopCreateRearrangeDescriptor(
    infiniopHandle_t handle,
    infiniopRearrangeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t dst,
    infiniopTensorDescriptor_t src) {
    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateRearrangeDescriptor(handle, (RearrangeCpuDescriptor_t *)desc_ptr, dst, src);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaCreateRearrangeDescriptor((CudaHandle_t)handle, (RearrangeCudaDescriptor_t *)desc_ptr, dst, src);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCreateRearrangeDescriptor((BangHandle_t)handle, (RearrangeBangDescriptor_t *)desc_ptr, dst, src);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return aclnnCreateRearrangeDescriptor((AscendHandle_t)handle,
                                              (RearrangeAclnnDescriptor_t *)desc_ptr,
                                              dst,
                                              src);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaCreateRearrangeDescriptor((MacaHandle_t)handle, (RearrangeMacaDescriptor_t *)desc_ptr, dst, src);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaCreateRearrangeDescriptor((MusaHandle_t)handle, (RearrangeMusaDescriptor_t *)desc_ptr, dst, src);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopRearrange(infiniopRearrangeDescriptor_t desc, void *dst, const void *src, void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuRearrange((RearrangeCpuDescriptor_t)desc, dst, src, stream);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaRearrange((RearrangeCudaDescriptor_t)desc, dst, src, stream);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangRearrange((RearrangeBangDescriptor_t)desc, dst, src, stream);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return aclnnRearrange((RearrangeAclnnDescriptor_t)desc,
                              dst,
                              src,
                              stream);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaRearrange((RearrangeMacaDescriptor_t)desc, dst, src, stream);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaRearrange((RearrangeMusaDescriptor_t)desc, dst, src, stream);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyRearrangeDescriptor(infiniopRearrangeDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyRearrangeDescriptor((RearrangeCpuDescriptor_t)desc);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaDestroyRearrangeDescriptor((RearrangeCudaDescriptor_t)desc);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangDestroyRearrangeDescriptor((RearrangeBangDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return aclnnDestroyRearrangeDescriptor((RearrangeAclnnDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaDestroyRearrangeDescriptor((RearrangeMacaDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaDestroyRearrangeDescriptor((RearrangeMusaDescriptor_t)desc);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
