#include "infinirt_bang.h"
#include "../../utils.h"
#include "cnrt.h"

#define CHECK_BANGRT(RT_API) CHECK_INTERNAL(RT_API, cnrtSuccess)

namespace infinirt::bang {
infiniStatus_t getDeviceCount(int *count) {
    unsigned int device_count = static_cast<unsigned int>(*count);
    CHECK_BANGRT(cnrtGetDeviceCount(&device_count));
    *count = static_cast<int>(device_count);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_BANGRT(cnrtSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_BANGRT(cnrtSyncDevice());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    cnrtQueue_t queue;
    CHECK_BANGRT(cnrtQueueCreate(&queue));
    *stream_ptr = queue;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_BANGRT(cnrtQueueDestroy((cnrtQueue_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_BANGRT(cnrtQueueSync((cnrtQueue_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_BANGRT(cnrtQueueWaitNotifier((cnrtNotifier_t)event, (cnrtQueue_t)stream, 0));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    cnrtNotifier_t notifier;
    CHECK_BANGRT(cnrtNotifierCreate(&notifier));
    *event_ptr = notifier;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_BANGRT(cnrtPlaceNotifier((cnrtNotifier_t)event, (cnrtQueue_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    auto status = cnrtQueryNotifier((cnrtNotifier_t)event);
    if (status == cnrtSuccess) {
        *status_ptr = INFINIRT_EVENT_COMPLETE;
    } else if (status == cnrtErrorBusy) {
        *status_ptr = INFINIRT_EVENT_NOT_READY;
    } else {
        CHECK_BANGRT(status);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_BANGRT(cnrtWaitNotifier((cnrtNotifier_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_BANGRT(cnrtNotifierDestroy((cnrtNotifier_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_BANGRT(cnrtMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_BANGRT(cnrtHostMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_BANGRT(cnrtFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_BANGRT(cnrtFreeHost(ptr));
    return INFINI_STATUS_SUCCESS;
}

cnrtMemTransDir_t toBangMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return cnrtMemcpyHostToDev;
    case INFINIRT_MEMCPY_D2H:
        return cnrtMemcpyDevToHost;
    case INFINIRT_MEMCPY_D2D:
        return cnrtMemcpyDevToDev;
    case INFINIRT_MEMCPY_H2H:
        return cnrtMemcpyHostToHost;
    default:
        return cnrtMemcpyNoDirection;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_BANGRT(cnrtMemcpy(dst, (void *)src, size, toBangMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_BANGRT(cnrtMemcpyAsync_V2(dst, (void *)src, size, (cnrtQueue_t)stream, toBangMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

// Does not support async malloc. Use blocking-style malloc instead
infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    CHECK_BANGRT(cnrtMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

// Does not support async free. Use blocking-style free instead
infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    CHECK_BANGRT(cnrtFree(ptr));
    return INFINI_STATUS_SUCCESS;
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

} // namespace infinirt::bang
