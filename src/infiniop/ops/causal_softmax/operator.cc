#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/causal_softmax.h"

#ifdef ENABLE_CPU_API
#include "cpu/causal_softmax_cpu.h"
#endif

__C infiniStatus_t infiniopCreateCausalSoftmaxDescriptor(
    infiniopHandle_t handle,
    infiniopCausalSoftmaxDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t y_desc) {

#define CREATE(CASE, NAMESPACE)                                                       \
    case CASE:                                                                        \
        return op::causal_softmax::NAMESPACE::Descriptor::create(                     \
            handle,                                                                   \
            reinterpret_cast<op::causal_softmax::NAMESPACE::Descriptor **>(desc_ptr), \
            y_desc);

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaCreateCausalSoftmaxDescriptor((CudaHandle_t)handle, (CausalSoftmaxCudaDescriptor_t *)desc_ptr, y_desc);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCreateCausalSoftmaxDescriptor((BangHandle_t)handle, (CausalSoftmaxBangDescriptor_t *)desc_ptr, y_desc);
        // return cnnlCreateCausalSoftmaxDescriptor((BangHandle_t) handle, (CausalSoftmaxCnnlDescriptor_t *) desc_ptr, y_desc);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return aclnnCreateCausalSoftmaxDescriptor((AscendHandle_t)handle, (CausalSoftmaxAclnnDescriptor_t *)desc_ptr, y_desc);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaCreateCausalSoftmaxDescriptor((MacaHandle_t)handle, (CausalSoftmaxMacaDescriptor_t *)desc_ptr, y_desc);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaCreateCausalSoftmaxDescriptor((MusaHandle_t)handle, (CausalSoftmaxMusaDescriptor_t *)desc_ptr, y_desc);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopGetCausalSoftmaxWorkspaceSize(infiniopCausalSoftmaxDescriptor_t desc, size_t *size) {

#define GET(CASE, NAMESPACE)                                                                          \
    case CASE:                                                                                        \
        *size = reinterpret_cast<op::causal_softmax::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaGetCausalSoftmaxWorkspaceSize((CausalSoftmaxCudaDescriptor_t)desc, size);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangGetCausalSoftmaxWorkspaceSize((CausalSoftmaxBangDescriptor_t)desc, size);
        // return cnnlGetCausalSoftmaxWorkspaceSize((CausalSoftmaxCnnlDescriptor_t) desc, size);
    }

#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return aclnnGetCausalSoftmaxWorkspaceSize((CausalSoftmaxAclnnDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaGetCausalSoftmaxWorkspaceSize((CausalSoftmaxMacaDescriptor_t)desc, size);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaGetCausalSoftmaxWorkspaceSize((CausalSoftmaxMusaDescriptor_t)desc, size);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopCausalSoftmax(infiniopCausalSoftmaxDescriptor_t desc, void *workspace, size_t workspace_size, void *data, void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                             \
    case CASE:                                                                                 \
        return reinterpret_cast<op::causal_softmax::NAMESPACE::Descriptor *>(desc)->calculate( \
            workspace, workspace_size, data, stream);

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaCausalSoftmax((CausalSoftmaxCudaDescriptor_t)desc, workspace, workspace_size, data, stream);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangCausalSoftmax((CausalSoftmaxBangDescriptor_t)desc, workspace, workspace_size, data, stream);
        // return cnnlCausalSoftmax((CausalSoftmaxCnnlDescriptor_t) desc, workspace, workspace_size, data, stream);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return aclnnCausalSoftmax((CausalSoftmaxAclnnDescriptor_t)desc, workspace, workspace_size, data, stream);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaCausalSoftmax((CausalSoftmaxMacaDescriptor_t)desc, workspace, workspace_size, data, stream);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu: {
        return musaCausalSoftmax((CausalSoftmaxMusaDescriptor_t)desc, workspace, workspace_size, data, stream);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyCausalSoftmaxDescriptor(infiniopCausalSoftmaxDescriptor_t desc) {

#define DESTROY(CASE, NAMESPACE)                                                    \
    case CASE:                                                                      \
        delete reinterpret_cast<op::causal_softmax::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {
#ifdef ENABLE_CPU_API
        DESTROY(INFINI_DEVICE_CPU, cpu)
#endif
#ifdef ENABLE_NV_GPU
    case DevNvGpu: {
        return cudaDestroyCausalSoftmaxDescriptor((CausalSoftmaxCudaDescriptor_t)desc);
    }

#endif
#ifdef ENABLE_CAMBRICON_MLU
    case DevCambriconMlu: {
        return bangDestroyCausalSoftmaxDescriptor((CausalSoftmaxBangDescriptor_t)desc);
        // return cnnlDestroyCausalSoftmaxDescriptor((CausalSoftmaxCnnlDescriptor_t) desc);
    }
#endif
#ifdef ENABLE_ASCEND_NPU
    case DevAscendNpu: {
        return aclnnDestroyCausalSoftmaxDescriptor((CausalSoftmaxAclnnDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_METAX_GPU
    case DevMetaxGpu: {
        return macaDestroyCausalSoftmaxDescriptor((CausalSoftmaxMacaDescriptor_t)desc);
    }
#endif
#ifdef ENABLE_MTHREADS_GPU
    case DevMthreadsGpu:
        return musaDestroyCausalSoftmaxDescriptor((CausalSoftmaxMusaDescriptor_t)desc);
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
