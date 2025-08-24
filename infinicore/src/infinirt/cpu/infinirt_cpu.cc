#include "infinirt_cpu.h"
#include <cstdlib>
#include <cstring>

namespace infinirt::cpu {
infiniStatus_t getDeviceCount(int *count) {
    *count = 1;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    *stream_ptr = nullptr;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    *p_ptr = std::malloc(size);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    return mallocDevice(p_ptr, size);
}

infiniStatus_t freeDevice(void *ptr) {
    std::free(ptr);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    return freeDevice(ptr);
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    std::memcpy(dst, src, size);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    return memcpy(dst, src, size, kind);
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    return mallocDevice(p_ptr, size);
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    return freeDevice(ptr);
}

} // namespace infinirt::cpu
