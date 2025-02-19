#include "infiniop/ops/swiglu.h"

__C infiniopStatus_t infiniopCreateSwiGLUDescriptor(
    infiniopHandle_t handle, infiniopSwiGLUDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc, infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc) {
    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateSwiGLUDescriptor(
            handle, (SwiGLUCpuDescriptor_t *)desc_ptr, c_desc, a_desc, b_desc);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaCreateSwiGLUDescriptor((CudaHandle_t)handle,
                                          (SwiGLUCudaDescriptor_t *)desc_ptr,
                                          c_desc, a_desc, b_desc);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCreateSwiGLUDescriptor((BangHandle_t)handle,
                                          (SwiGLUBangDescriptor_t *)desc_ptr,
                                          c_desc, a_desc, b_desc);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu:
        return ascendCreateSwiGLUDescriptor(
            (AscendHandle_t)handle, (SwiGLUAscendDescriptor_t *)desc_ptr,
            c_desc, a_desc, b_desc);
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaCreateSwiGLUDescriptor((MacaHandle_t)handle,
                                          (SwiGLUMacaDescriptor_t *)desc_ptr,
                                          c_desc, a_desc, b_desc);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu:
        return musaCreateSwiGLUDescriptor(
            handle, (SwiGLUMusaDescriptor_t *)desc_ptr, c_desc, a_desc, b_desc);
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
};

__C infiniopStatus_t infiniopSwiGLU(infiniopSwiGLUDescriptor_t desc, void *c,
                                    void const *a, void const *b,
                                    void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuSwiGLU((SwiGLUCpuDescriptor_t)desc, c, a, b, stream);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaSwiGLU((SwiGLUCudaDescriptor_t)desc, c, a, b, stream);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangSwiGLU((SwiGLUBangDescriptor_t)desc, c, a, b, stream);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu:
        return ascendSwiGLU((SwiGLUAscendDescriptor_t)desc, c, a, b, stream);
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu:
        return macaSwiGLU((SwiGLUMacaDescriptor_t)desc, c, a, b, stream);
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu:
        return musaSwiGLU((SwiGLUMusaDescriptor_t)desc, c, a, b, stream);
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t
infiniopDestroySwiGLUDescriptor(infiniopSwiGLUDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroySwiGLUDescriptor((SwiGLUCpuDescriptor_t)desc);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaDestroySwiGLUDescriptor((SwiGLUCudaDescriptor_t)desc);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangDestroySwiGLUDescriptor((SwiGLUBangDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu:
        return ascendDestroySwiGLUDescriptor((SwiGLUAscendDescriptor_t)desc);
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu:
        return macaDestroySwiGLUDescriptor((SwiGLUMacaDescriptor_t)desc);
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu:
        return musaDestroySwiGLUDescriptor((SwiGLUMusaDescriptor_t)desc);
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
