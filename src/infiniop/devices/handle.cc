#include "infiniop/handle.h"
#ifdef ENABLE_CPU_API
#include "cpu/cpu_handle.h"
#endif
#ifdef ENABLE_CUDA_API
#include "cuda/cuda_handle.h"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/bang_handle.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/ascend_handle.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "./kunlun/kunlun_handle.h"
#endif

__C infiniStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr,
                                        infiniDevice_t device) {
    if (handle_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    switch (device) {
#ifdef ENABLE_CPU_API
    case INFINI_DEVICE_CPU:
        return createCpuHandle((infiniopCpuHandle_t *)handle_ptr);
#endif
#ifdef ENABLE_CUDA_API
    case INFINI_DEVICE_NVIDIA: {
        return createCudaHandle((infiniopCudaHandle_t *)handle_ptr, device);
    }
#endif
#ifdef ENABLE_CAMBRICON_API
    case INFINI_DEVICE_CAMBRICON: {
        return createBangHandle((infiniopBangHandle_t *)handle_ptr);
    }
#endif
#ifdef ENABLE_ASCEND_API
    case INFINI_DEVICE_ASCEND: {
        return createAscendHandle((infiniopAscendHandle_t *)handle_ptr);
    }
#endif
#ifdef ENABLE_KUNLUN_API
    case INFINI_DEVICE_KUNLUN: {
        return createKunlunHandle((infiniopKunlunHandle_t *)handle_ptr);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

__C infiniStatus_t infiniopDestroyHandle(infiniopHandle_t handle) {
    switch (handle->device) {
#ifdef ENABLE_CPU_API
    case INFINI_DEVICE_CPU:
        return destroyCpuHandle((infiniopCpuHandle_t)handle);
#endif
#ifdef ENABLE_CUDA_API
    case INFINI_DEVICE_NVIDIA: {
        return destroyCudaHandle((infiniopCudaHandle_t)handle);
    }
#endif
#ifdef ENABLE_CAMBRICON_API
    case INFINI_DEVICE_CAMBRICON: {
        return destroyBangHandle((infiniopBangHandle_t)handle);
    }
#endif
#ifdef ENABLE_ASCEND_API
    case INFINI_DEVICE_ASCEND: {
        return destroyAscendHandle((infiniopAscendHandle_t)handle);
    }
#endif
#ifdef ENABLE_KUNLUN_API
    case INFINI_DEVICE_KUNLUN: {
        return destroyKunlunHandle((infiniopKunlunHandle_t)handle);
    }
#endif
    }
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}
