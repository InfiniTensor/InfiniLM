#ifndef __INFINIRT_IMPL_H__
#define __INFINIRT_IMPL_H__
#include "infinirt.h"

#define INFINIRT_DEVICE_API(IMPL, COUNT)                                                                                          \
    infiniStatus_t getDeviceCount(int *count) COUNT;                                                                              \
    infiniStatus_t setDevice(int device_id) IMPL;                                                                                 \
    infiniStatus_t deviceSynchronize() IMPL;                                                                                      \
                                                                                                                                  \
    infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) IMPL;                                                               \
    infiniStatus_t streamDestroy(infinirtStream_t stream) IMPL;                                                                   \
    infiniStatus_t streamSynchronize(infinirtStream_t stream) IMPL;                                                               \
    infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) IMPL;                                          \
                                                                                                                                  \
    infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) IMPL;                                                                  \
    infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) IMPL;                                              \
    infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) IMPL;                                     \
    infiniStatus_t eventSynchronize(infinirtEvent_t event) IMPL;                                                                  \
    infiniStatus_t eventDestroy(infinirtEvent_t event) IMPL;                                                                      \
                                                                                                                                  \
    infiniStatus_t mallocDevice(void **p_ptr, size_t size) IMPL;                                                                  \
    infiniStatus_t mallocHost(void **p_ptr, size_t size) IMPL;                                                                    \
    infiniStatus_t freeDevice(void *ptr) IMPL;                                                                                    \
    infiniStatus_t freeHost(void *ptr) IMPL;                                                                                      \
                                                                                                                                  \
    infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) IMPL;                               \
    infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) IMPL; \
                                                                                                                                  \
    infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) IMPL;                                          \
    infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) IMPL;

#define INFINIRT_DEVICE_API_IMPL INFINIRT_DEVICE_API(, )
#define INFINIRT_DEVICE_API_NOOP INFINIRT_DEVICE_API({ return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED; }, \
                                                     {*count = 0; return INFINI_STATUS_SUCCESS; })

#endif // __INFINIRT_IMPL_H__
