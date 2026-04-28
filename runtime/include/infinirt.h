#ifndef __INFINIRT_API_H__
#define __INFINIRT_API_H__

#include "infinicore.h"
#include <stdint.h>

typedef void *infinirtStream_t;
typedef void *infinirtEvent_t;
typedef void *infinirtGraph_t;
typedef void *infinirtGraphNode_t;
typedef void *infinirtGraphExec_t;

__INFINI_C __export infiniStatus_t infinirtInit();

// Device
__INFINI_C __export infiniStatus_t infinirtGetAllDeviceCount(int *count_array);
__INFINI_C __export infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count);
__INFINI_C __export infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id);
__INFINI_C __export infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr);
__INFINI_C __export infiniStatus_t infinirtDeviceSynchronize();

// Stream
__INFINI_C __export infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr);
__INFINI_C __export infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

// Event
typedef enum {
    INFINIRT_EVENT_COMPLETE = 0,
    INFINIRT_EVENT_NOT_READY = 1,
} infinirtEventStatus_t;

// Event flags for precise timing
typedef enum {
    INFINIRT_EVENT_DEFAULT = 0x0,        // Default event creation flags
    INFINIRT_EVENT_DISABLE_TIMING = 0x1, // Event will not record timing data
    INFINIRT_EVENT_BLOCKING_SYNC = 0x2,  // Event uses blocking synchronization
} infinirtEventFlags_t;

__INFINI_C __export infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr);
__INFINI_C __export infiniStatus_t infinirtEventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags);
__INFINI_C __export infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr);
__INFINI_C __export infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event);
__INFINI_C __export infiniStatus_t infinirtEventDestroy(infinirtEvent_t event);
__INFINI_C __export infiniStatus_t infinirtEventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end);

// Memory
typedef enum {
    INFINIRT_MEMCPY_H2H = 0,
    INFINIRT_MEMCPY_H2D = 1,
    INFINIRT_MEMCPY_D2H = 2,
    INFINIRT_MEMCPY_D2D = 3,
} infinirtMemcpyKind_t;

__INFINI_C __export infiniStatus_t infinirtMalloc(void **p_ptr, size_t size);
__INFINI_C __export infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size);
__INFINI_C __export infiniStatus_t infinirtFree(void *ptr);
__INFINI_C __export infiniStatus_t infinirtFreeHost(void *ptr);

__INFINI_C __export infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind);
__INFINI_C __export infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream);

// Stream-ordered memory
__INFINI_C __export infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream);
__INFINI_C __export infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream);

// Graph
typedef enum {
    INFINIRT_STREAM_CAPTURE_MODE_GLOBAL = 0,
    INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    INFINIRT_STREAM_CAPTURE_MODE_RELAXED = 2,

} infinirtStreamCaptureMode_t;

__INFINI_C __export infiniStatus_t infinirtStreamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode);
__INFINI_C __export infiniStatus_t infinirtStreamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr);
__INFINI_C __export infiniStatus_t infinirtGraphDestroy(infinirtGraph_t graph);
__INFINI_C __export infiniStatus_t infinirtGraphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size);
__INFINI_C __export infiniStatus_t infinirtGraphExecDestroy(infinirtGraphExec_t graph_exec);
__INFINI_C __export infiniStatus_t infinirtGraphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream);

#endif // __INFINIRT_API_H__
