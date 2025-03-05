#include "infiniop/handle.h"
#include "../../utils.h"
#include "infinirt.h"

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
#include "kunlun/kunlun_handle.h"
#endif

__C infiniStatus_t infiniopCreateHandle(infiniopHandle_t *handle_ptr) {
    if (handle_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    infiniDevice_t device;
    int device_id;
    CHECK_STATUS(infinirtGetDevice(&device, &device_id));

#define CREATE(CASE, NAMESPACE) \
    case CASE:                  \
        return device::NAMESPACE::Handle::create(handle_ptr, device_id)

    switch (device) {
#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        CREATE(INFINI_DEVICE_NVIDIA, cuda::nvidia);
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

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t infiniopDestroyHandle(infiniopHandle_t handle) {

#define DELETE(CASE, NAMESPACE)                                       \
    case CASE:                                                        \
        delete reinterpret_cast<device::NAMESPACE::Handle *>(handle); \
        return INFINI_STATUS_SUCCESS

    switch (handle->device) {
#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_CUDA_API
        DELETE(INFINI_DEVICE_NVIDIA, cuda::nvidia);
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
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}
