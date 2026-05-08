#include "infinirt_cpu.h"
#include <chrono>
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
    // For CPU implementation, we use a simple timestamp as event
    auto now = std::chrono::steady_clock::now();
    auto *timestamp = new std::chrono::steady_clock::time_point(now);
    *event_ptr = timestamp;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    // CPU implementation ignores flags for simplicity
    return eventCreate(event_ptr);
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    // Update the event timestamp
    auto *timestamp = static_cast<std::chrono::steady_clock::time_point *>(event);
    *timestamp = std::chrono::steady_clock::now();
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    // CPU events are always complete immediately
    *status_ptr = INFINIRT_EVENT_COMPLETE;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    // CPU events are synchronized immediately
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    auto *timestamp = static_cast<std::chrono::steady_clock::time_point *>(event);
    delete timestamp;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    auto *start_time = static_cast<std::chrono::steady_clock::time_point *>(start);
    auto *end_time = static_cast<std::chrono::steady_clock::time_point *>(end);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(*end_time - *start_time);
    *ms_ptr = static_cast<float>(duration.count()) / 1000.0f; // Convert microseconds to milliseconds

    return INFINI_STATUS_SUCCESS;
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

infiniStatus_t streamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t streamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphDestroy(infinirtGraph_t graph) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphExecDestroy(infinirtGraphExec_t graph_exec) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

infiniStatus_t graphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
}

} // namespace infinirt::cpu
