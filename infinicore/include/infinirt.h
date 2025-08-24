#ifndef __INFINIRT_API_H__
#define __INFINIRT_API_H__

#include "infinicore.h"

typedef void *infinirtStream_t;
typedef void *infinirtEvent_t;

__C __export infiniStatus_t infinirtInit();

// Device
__C __export infiniStatus_t infinirtGetAllDeviceCount(int *count_array);
__C __export infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count);
__C __export infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id);
__C __export infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr);
__C __export infiniStatus_t infinirtDeviceSynchronize();

// Stream
__C __export infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr);
__C __export infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream);
__C __export infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream);
__C __export infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event);

// Event
typedef enum {
    INFINIRT_EVENT_COMPLETE = 0,
    INFINIRT_EVENT_NOT_READY = 1,
} infinirtEventStatus_t;

__C __export infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr);
__C __export infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream);
__C __export infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr);
__C __export infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event);
__C __export infiniStatus_t infinirtEventDestroy(infinirtEvent_t event);

// Memory
typedef enum {
    INFINIRT_MEMCPY_H2H = 0,
    INFINIRT_MEMCPY_H2D = 1,
    INFINIRT_MEMCPY_D2H = 2,
    INFINIRT_MEMCPY_D2D = 3,
} infinirtMemcpyKind_t;

__C __export infiniStatus_t infinirtMalloc(void **p_ptr, size_t size);
__C __export infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size);
__C __export infiniStatus_t infinirtFree(void *ptr);
__C __export infiniStatus_t infinirtFreeHost(void *ptr);

__C __export infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind);
__C __export infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream);

// Stream-ordered memory
__C __export infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream);
__C __export infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream);

#endif // __INFINIRT_API_H__
