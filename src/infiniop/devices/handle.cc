#include "infiniop/handle.h"
#ifdef ENABLE_CPU_API
#include "./cpu/cpu_handle.h"
#endif
#ifdef ENABLE_CUDA_API
#include "./cuda/cuda_handle.h"
#endif
#ifdef ENABLE_CAMBRICON_MLU
#include "./bang/bang_handle.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "./ascend/ascend_handle.h"
#endif


__C infiniopStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr, infiniDevice_t device, int device_id) {
    if (handle_ptr == nullptr) {
        return INFINIOP_STATUS_NULL_POINTER;
    }
    if (device_id < 0) {
        return INFINIOP_STATUS_BAD_DEVICE;
    }

    switch (device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            return createCpuHandle((infiniopCpuHandle_t *) handle_ptr);
#endif
#ifdef ENABLE_CUDA_API
        case INFINI_DEVICE_NVIDIA: {
            return createCudaHandle((infiniopCudaHandle_t *) handle_ptr, device_id, device);
        }
#endif
#ifdef ENABLE_CAMBRICON_API
        case DevCambriconMlu: {
            return createBangHandle((infiniopBangHandle_t *) handle_ptr, device_id);
        }
#endif
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND: {
            return createAscendHandle((infiniopAscendHandle_t *) handle_ptr, device_id);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}


__C infiniopStatus_t infiniopDestroyHandle(infiniopHandle_t handle) {
    switch (handle->device) {
#ifdef ENABLE_CPU_API
        case INFINI_DEVICE_CPU:
            delete handle;
            return INFINIOP_STATUS_SUCCESS;
#endif
#ifdef ENABLE_CUDA_API
        case INFINI_DEVICE_NVIDIA: {
            return deleteCudaHandle((infiniopCudaHandle_t) handle);
        }
#endif
#ifdef ENABLE_CAMBRICON_MLU
        case DevCambriconMlu: {
            delete (infiniopBangHandle_t) handle;
            return STATUS_SUCCESS;
        }
#endif
#ifdef ENABLE_ASCEND_API
        case INFINI_DEVICE_ASCEND: {
            return deleteAscendHandle((infiniopAscendHandle_t) handle);
        }
#endif
    }
    return INFINIOP_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}