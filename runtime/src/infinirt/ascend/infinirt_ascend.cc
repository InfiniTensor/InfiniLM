#include "infinirt_ascend.h"
#include "../../utils.h"
#include <acl/acl.h>
#include <mutex>

std::once_flag acl_init_flag;

#define CHECK_ACLRT(API) CHECK_INTERNAL(API, ACL_SUCCESS)

namespace infinirt::ascend {

infiniStatus_t init() {
    aclError _err = ACL_SUCCESS;
    std::call_once(acl_init_flag, [&_err]() {
        _err = aclInit(NULL);
    });
    CHECK_ACLRT(_err);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t getDeviceCount(int *count) {
    uint32_t count_ = 0;
    CHECK_ACLRT(aclrtGetDeviceCount(&count_));
    *count = (int)count_;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_ACLRT(aclrtSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_ACLRT(aclrtSynchronizeDevice());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    aclrtStream acl_stream;
    CHECK_ACLRT(aclrtCreateStreamWithConfig(&acl_stream, 0, ACL_STREAM_FAST_LAUNCH));
    *stream_ptr = (infinirtStream_t)acl_stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    CHECK_ACLRT(aclrtDestroyStream((aclrtStream)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_ACLRT(aclrtSynchronizeStream((aclrtStream)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    CHECK_ACLRT(aclrtStreamWaitEvent((aclrtStream)stream, (aclrtEvent)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    aclrtEvent acl_event;
    CHECK_ACLRT(aclrtCreateEvent(&acl_event));
    *event_ptr = (infinirtEvent_t)acl_event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_ACLRT(aclrtRecordEvent((aclrtEvent)event, (aclrtStream)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    aclrtEventRecordedStatus status;
    CHECK_ACLRT(aclrtQueryEventStatus((aclrtEvent)event, &status));
    if (ACL_EVENT_RECORDED_STATUS_COMPLETE == status) {
        *status_ptr = INFINIRT_EVENT_COMPLETE;
    } else {
        *status_ptr = INFINIRT_EVENT_NOT_READY;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_ACLRT(aclrtSynchronizeEvent((aclrtEvent)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    CHECK_ACLRT(aclrtDestroyEvent((aclrtEvent)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    return INFINI_STATUS_NOT_IMPLEMENTED;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_ACLRT(aclrtMallocAlign32(p_ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_ACLRT(aclrtMallocHost(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    CHECK_ACLRT(aclrtFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    CHECK_ACLRT(aclrtFreeHost(ptr));
    return INFINI_STATUS_SUCCESS;
}

aclrtMemcpyKind toAclrtMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return ACL_MEMCPY_HOST_TO_DEVICE;
    case INFINIRT_MEMCPY_D2H:
        return ACL_MEMCPY_DEVICE_TO_HOST;
    case INFINIRT_MEMCPY_D2D:
        return ACL_MEMCPY_DEVICE_TO_DEVICE;
    case INFINIRT_MEMCPY_H2H:
        return ACL_MEMCPY_HOST_TO_HOST;
    default:
        return ACL_MEMCPY_DEFAULT;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_ACLRT(aclrtMemcpy(dst, size, src, size, toAclrtMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_ACLRT(aclrtMemcpyAsync(dst, size, src, size, toAclrtMemcpyKind(kind), (aclrtStream)stream));
    return INFINI_STATUS_SUCCESS;
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

} // namespace infinirt::ascend
#undef CHECK_ACLRT
