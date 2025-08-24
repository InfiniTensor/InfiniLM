#include "infinirt_kunlun.h"
#include "../../utils.h"
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>

typedef XPUStream kunlunStream_t;
typedef XPUEvent kunlunEvent_t;

#define CHECK_KUNLUNRT(RT_API) CHECK_INTERNAL(RT_API, XPU_SUCCESS)

namespace infinirt::kunlun {
infiniStatus_t getDeviceCount(int *count) {
    CHECK_KUNLUNRT(xpu_device_count(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_KUNLUNRT(xpu_set_device(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_KUNLUNRT(xpu_wait());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    kunlunStream_t stream;
    CHECK_KUNLUNRT(xpu_stream_create(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_stream_destroy((kunlunStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_wait((kunlunStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}
infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_KUNLUNRT(xpu_stream_wait_event((kunlunStream_t)stream, (kunlunEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    kunlunEvent_t event;
    CHECK_KUNLUNRT(xpu_event_create(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_event_record((kunlunEvent_t)event, (kunlunStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    // no event query in kunlun2
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_KUNLUNRT(xpu_event_wait((kunlunEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_KUNLUNRT(xpu_event_destroy((kunlunEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_KUNLUNRT(xpu_malloc(p_ptr, static_cast<uint64_t>(size)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_KUNLUNRT(xpu_host_alloc(p_ptr, static_cast<uint64_t>(size), 0));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_KUNLUNRT(xpu_free(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_KUNLUNRT(xpu_host_free(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        CHECK_KUNLUNRT(xpu_memcpy(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_HOST_TO_DEVICE));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_D2H:
        CHECK_KUNLUNRT(xpu_memcpy(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_DEVICE_TO_HOST));
        return INFINI_STATUS_SUCCESS;
    case INFINIRT_MEMCPY_D2D:
        CHECK_KUNLUNRT(xpu_memcpy(dst, src, static_cast<uint64_t>(size), XPUMemcpyKind::XPU_DEVICE_TO_DEVICE));
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_INTERNAL_ERROR;
    }
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    // no async memcpy func in kunlun2
    return memcpy(dst, src, size, kind);
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_malloc(p_ptr, static_cast<uint64_t>(size)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_KUNLUNRT(xpu_free(ptr));
    return INFINI_STATUS_SUCCESS;
}

} // namespace infinirt::kunlun
