#include "infiniop/ops/random_sample.h"

__C infiniopStatus_t infiniopCreateRandomSampleDescriptor(infiniopHandle_t handle, infiniopRandomSampleDescriptor_t *desc_ptr, infiniopTensorDescriptor_t result, infiniopTensorDescriptor_t probs) {
    switch (handle->device) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuCreateRandomSampleDescriptor(handle, (RandomSampleCpuDescriptor_t *)desc_ptr, result, probs);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaCreateRandomSampleDescriptor((CudaHandle_t)handle, (RandomSampleCudaDescriptor_t *)desc_ptr, result, probs);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCreateRandomSampleDescriptor((BangHandle_t)handle,
                                                (RandomSampleBangDescriptor_t *)desc_ptr, result,
                                                probs);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendCreateRandomSampleDescriptor((AscendHandle_t)handle,
                                                  (RandomSampleAscendDescriptor_t *)desc_ptr, result, probs);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaCreateRandomSampleDescriptor((MacaHandle_t)handle,
                                                (RandomSampleMacaDescriptor_t *)desc_ptr, result,
                                                probs);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu:
        return musaCreateRandomSampleDescriptor((MusaHandle_t)handle, (RandomSampleMusaDescriptor_t *)desc_ptr, result, probs);
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
};

__C infiniopStatus_t infiniopGetRandomSampleWorkspaceSize(infiniopRandomSampleDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuGetRandomSampleWorkspaceSize((RandomSampleCpuDescriptor_t)desc, size);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaGetRandomSampleWorkspaceSize((RandomSampleCudaDescriptor_t)desc, size);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangGetRandomSampleWorkspaceSize((RandomSampleBangDescriptor_t)desc, size);
        // return cnnlGetRandomSampleWorkspaceSize((RandomSampleCnnlDescriptor_t) desc, size);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendGetRandomSampleWorkspaceSize((RandomSampleAscendDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaGetRandomSampleWorkspaceSize((RandomSampleMacaDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaGetRandomSampleWorkspaceSize((RandomSampleMusaDescriptor_t)desc, size);
    }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopRandomSample(infiniopRandomSampleDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *result,
                                          const void *probs,
                                          float random_val,
                                          float topp,
                                          int topk,
                                          float temperature,
                                          void *stream) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuRandomSample((RandomSampleCpuDescriptor_t)desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaRandomSample((RandomSampleCudaDescriptor_t)desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangRandomSample((RandomSampleBangDescriptor_t)desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendRandomSample((RandomSampleAscendDescriptor_t)desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaRandomSample((RandomSampleMacaDescriptor_t)desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu:
        return musaRandomSample((RandomSampleMusaDescriptor_t)desc, workspace, workspace_size, result, probs, random_val, topp, topk, temperature, stream);
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniopStatus_t infiniopDestroyRandomSampleDescriptor(infiniopRandomSampleDescriptor_t desc) {
    switch (desc->device_type) {
#ifdef ENABLE_CPU
    case DevCpu:
        return cpuDestroyRandomSampleDescriptor((RandomSampleCpuDescriptor_t)desc);
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu:
        return cudaDestroyRandomSampleDescriptor((RandomSampleCudaDescriptor_t)desc);
#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangDestroyRandomSampleDescriptor((RandomSampleBangDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return ascendDestroyRandomSampleDescriptor((RandomSampleAscendDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaDestroyRandomSampleDescriptor((RandomSampleMacaDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu:
        return musaDestroyRandomSampleDescriptor((RandomSampleMusaDescriptor_t)desc);
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
