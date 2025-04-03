#include "infinirt_musa.h"
#include "../../utils.h"
#include <musa_runtime.h>
#include <musa_runtime_api.h>

#define CHECK_MUSART(RT_API) CHECK_INTERNAL(RT_API, musaSuccess)

namespace infinirt::musa {
infiniStatus_t getDeviceCount(int *count) {
    CHECK_MUSART(musaGetDeviceCount(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_MUSART(musaSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_MUSART(musaDeviceSynchronize());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    musaStream_t stream;
    CHECK_MUSART(musaStreamCreate(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_MUSART(musaStreamDestroy((musaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_MUSART(musaStreamSynchronize((musaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_MUSART(musaStreamWaitEvent((musaStream_t)stream, (musaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    musaEvent_t event;
    CHECK_MUSART(musaEventCreate(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_MUSART(musaEventRecord((musaEvent_t)event, (musaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    auto status = musaEventQuery((musaEvent_t)event);
    if (status == musaSuccess) {
        *status_ptr = INFINIRT_EVENT_COMPLETE;
    } else if (status == musaErrorNotReady) {
        *status_ptr = INFINIRT_EVENT_NOT_READY;
    } else {
        CHECK_MUSART(status);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_MUSART(musaEventSynchronize((musaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_MUSART(musaEventDestroy((musaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_MUSART(musaMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_MUSART(musaMallocHost(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_MUSART(musaFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_MUSART(musaFreeHost(ptr));
    return INFINI_STATUS_SUCCESS;
}

musaMemcpyKind toMusaMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return musaMemcpyHostToDevice;
    case INFINIRT_MEMCPY_D2H:
        return musaMemcpyDeviceToHost;
    case INFINIRT_MEMCPY_D2D:
        return musaMemcpyDeviceToDevice;
    case INFINIRT_MEMCPY_H2H:
        return musaMemcpyHostToHost;
    default:
        return musaMemcpyDefault;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_MUSART(musaMemcpy(dst, src, size, toMusaMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_MUSART(musaMemcpyAsync(dst, src, size, toMusaMemcpyKind(kind), (musaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    return mallocDevice(p_ptr, size);
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return freeDevice(ptr);
}
} // namespace infinirt::musa
