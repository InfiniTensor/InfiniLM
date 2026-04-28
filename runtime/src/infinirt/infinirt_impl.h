#ifndef __INFINIRT_IMPL_H__
#define __INFINIRT_IMPL_H__
#include "infinirt.h"
#include <stdint.h>

#define INFINIRT_DEVICE_API(INLINE, IMPL, COUNT)                                                                                         \
    INLINE infiniStatus_t getDeviceCount(int *count) COUNT;                                                                              \
    INLINE infiniStatus_t setDevice(int device_id) IMPL;                                                                                 \
    INLINE infiniStatus_t deviceSynchronize() IMPL;                                                                                      \
                                                                                                                                         \
    INLINE infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) IMPL;                                                               \
    INLINE infiniStatus_t streamDestroy(infinirtStream_t stream) IMPL;                                                                   \
    INLINE infiniStatus_t streamSynchronize(infinirtStream_t stream) IMPL;                                                               \
    INLINE infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) IMPL;                                          \
                                                                                                                                         \
    INLINE infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) IMPL;                                                                  \
    INLINE infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) IMPL;                                         \
    INLINE infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) IMPL;                                              \
    INLINE infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) IMPL;                                     \
    INLINE infiniStatus_t eventSynchronize(infinirtEvent_t event) IMPL;                                                                  \
    INLINE infiniStatus_t eventDestroy(infinirtEvent_t event) IMPL;                                                                      \
    INLINE infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) IMPL;                              \
                                                                                                                                         \
    INLINE infiniStatus_t mallocDevice(void **p_ptr, size_t size) IMPL;                                                                  \
    INLINE infiniStatus_t mallocHost(void **p_ptr, size_t size) IMPL;                                                                    \
    INLINE infiniStatus_t freeDevice(void *ptr) IMPL;                                                                                    \
    INLINE infiniStatus_t freeHost(void *ptr) IMPL;                                                                                      \
                                                                                                                                         \
    INLINE infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) IMPL;                               \
    INLINE infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) IMPL; \
                                                                                                                                         \
    INLINE infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) IMPL;                                          \
    INLINE infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) IMPL;                                                            \
                                                                                                                                         \
    INLINE infiniStatus_t streamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) IMPL;                            \
    INLINE infiniStatus_t streamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) IMPL;                                    \
    INLINE infiniStatus_t graphDestroy(infinirtGraph_t graph) IMPL;                                                                      \
    INLINE infiniStatus_t graphInstantiate(                                                                                              \
        infinirtGraphExec_t *graph_exec_ptr,                                                                                             \
        infinirtGraph_t graph,                                                                                                           \
        infinirtGraphNode_t *node_ptr,                                                                                                   \
        char *log_buffer,                                                                                                                \
        size_t buffer_size) IMPL;                                                                                                        \
    INLINE infiniStatus_t graphExecDestroy(infinirtGraphExec_t graph_exec) IMPL;                                                         \
    INLINE infiniStatus_t graphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) IMPL;

#define INFINIRT_DEVICE_API_IMPL INFINIRT_DEVICE_API(, , )
#define INFINIRT_DEVICE_API_NOOP INFINIRT_DEVICE_API(            \
    inline, { return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED; }, \
    {*count = 0; return INFINI_STATUS_SUCCESS; })

#endif // __INFINIRT_IMPL_H__
