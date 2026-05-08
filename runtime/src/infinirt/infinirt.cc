#include "infinirt.h"
#include "../utils.h"
#include "ascend/infinirt_ascend.h"
#include "bang/infinirt_bang.h"
#include "cpu/infinirt_cpu.h"
#include "cuda/infinirt_cuda.cuh"
#include "kunlun/infinirt_kunlun.h"
#include "metax/infinirt_metax.h"
#include "moore/infinirt_moore.h"

thread_local infiniDevice_t CURRENT_DEVICE_TYPE = INFINI_DEVICE_CPU;
thread_local int CURRENT_DEVICE_ID = 0;
thread_local infiniDevice_t PREVIOUS_NON_CPU_DEVICE_TYPE = INFINI_DEVICE_TYPE_COUNT;
thread_local int PREVIOUS_NON_CPU_DEVICE_ID = 0;

__INFINI_C infiniStatus_t infinirtInit() {
#ifdef ENABLE_ASCEND_API
    CHECK_STATUS(infinirt::ascend::init());
#endif
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infinirtGetAllDeviceCount(int *count_array) {
    if (count_array == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    for (size_t i = 0; i < INFINI_DEVICE_TYPE_COUNT; i++) {
        auto status = infinirtGetDeviceCount(static_cast<infiniDevice_t>(i), &count_array[i]);
        if (status != INFINI_STATUS_SUCCESS) {
            return status;
        }
    }
    return INFINI_STATUS_SUCCESS;
}

__INFINI_C infiniStatus_t infinirtGetDevice(infiniDevice_t *device_ptr, int *device_id_ptr) {
    if (device_ptr == nullptr && device_id_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    if (device_ptr != nullptr) {
        *device_ptr = CURRENT_DEVICE_TYPE;
    }
    if (device_id_ptr != nullptr) {
        *device_id_ptr = CURRENT_DEVICE_ID;
    }

    return INFINI_STATUS_SUCCESS;
}

#define INFINIRT_CALL_DEVICE_API_AND(DEVICE_TYPE, API, PARAMS, ACTION) \
    {                                                                  \
        infiniStatus_t _status;                                        \
        switch (DEVICE_TYPE) {                                         \
        case INFINI_DEVICE_CPU:                                        \
            _status = infinirt::cpu::API PARAMS;                       \
            break;                                                     \
        case INFINI_DEVICE_NVIDIA:                                     \
            _status = infinirt::cuda::API PARAMS;                      \
            break;                                                     \
        case INFINI_DEVICE_CAMBRICON:                                  \
            _status = infinirt::bang::API PARAMS;                      \
            break;                                                     \
        case INFINI_DEVICE_ASCEND:                                     \
            _status = infinirt::ascend::API PARAMS;                    \
            break;                                                     \
        case INFINI_DEVICE_METAX:                                      \
            _status = infinirt::metax::API PARAMS;                     \
            break;                                                     \
        case INFINI_DEVICE_MOORE:                                      \
            _status = infinirt::musa::API PARAMS;                      \
            break;                                                     \
        case INFINI_DEVICE_KUNLUN:                                     \
            _status = infinirt::kunlun::API PARAMS;                    \
            break;                                                     \
        case INFINI_DEVICE_ILUVATAR:                                   \
            _status = infinirt::iluvatar::API PARAMS;                  \
            break;                                                     \
        case INFINI_DEVICE_QY:                                         \
            _status = infinirt::qy::API PARAMS;                        \
            break;                                                     \
        case INFINI_DEVICE_HYGON:                                      \
            _status = infinirt::hygon::API PARAMS;                     \
            break;                                                     \
        case INFINI_DEVICE_ALI:                                        \
            _status = infinirt::ali::API PARAMS;                       \
            break;                                                     \
        default:                                                       \
            _status = INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;         \
        }                                                              \
        { ACTION; }                                                    \
        return _status;                                                \
    }

#define INFINIRT_CALL_DEVICE_API(API, PARAMS) INFINIRT_CALL_DEVICE_API_AND(CURRENT_DEVICE_TYPE, API, PARAMS, )

__INFINI_C infiniStatus_t infinirtGetDeviceCount(infiniDevice_t device, int *count) {
    if (count == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    INFINIRT_CALL_DEVICE_API_AND(device, getDeviceCount, (count), {});
}

__INFINI_C infiniStatus_t infinirtSetDevice(infiniDevice_t device, int device_id) {
    bool skip_set = CURRENT_DEVICE_TYPE == INFINI_DEVICE_CPU && device == PREVIOUS_NON_CPU_DEVICE_TYPE && device_id == PREVIOUS_NON_CPU_DEVICE_ID;
    if (CURRENT_DEVICE_TYPE != INFINI_DEVICE_CPU) {
        PREVIOUS_NON_CPU_DEVICE_TYPE = CURRENT_DEVICE_TYPE;
        PREVIOUS_NON_CPU_DEVICE_ID = CURRENT_DEVICE_ID;
    }
    if (skip_set) {
        CURRENT_DEVICE_TYPE = device;
        CURRENT_DEVICE_ID = device_id;
        return INFINI_STATUS_SUCCESS;
    }
    INFINIRT_CALL_DEVICE_API_AND(device, setDevice, (device_id),
                                 { CURRENT_DEVICE_TYPE = device;
                                   CURRENT_DEVICE_ID = device_id; });
}

__INFINI_C infiniStatus_t infinirtDeviceSynchronize() {
    INFINIRT_CALL_DEVICE_API(deviceSynchronize, ());
}

__INFINI_C infiniStatus_t infinirtStreamCreate(infinirtStream_t *stream_ptr) {
    INFINIRT_CALL_DEVICE_API(streamCreate, (stream_ptr));
}

__INFINI_C infiniStatus_t infinirtStreamDestroy(infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(streamDestroy, (stream));
}

__INFINI_C infiniStatus_t infinirtStreamSynchronize(infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(streamSynchronize, (stream));
}

__INFINI_C infiniStatus_t infinirtStreamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
    INFINIRT_CALL_DEVICE_API(streamWaitEvent, (stream, event));
}

__INFINI_C infiniStatus_t infinirtEventCreate(infinirtEvent_t *event_ptr) {
    INFINIRT_CALL_DEVICE_API(eventCreate, (event_ptr));
}

__INFINI_C infiniStatus_t infinirtEventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    INFINIRT_CALL_DEVICE_API(eventCreateWithFlags, (event_ptr, flags));
}

__INFINI_C infiniStatus_t infinirtEventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(eventRecord, (event, stream));
}

__INFINI_C infiniStatus_t infinirtEventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    INFINIRT_CALL_DEVICE_API(eventQuery, (event, status_ptr));
}

__INFINI_C infiniStatus_t infinirtEventSynchronize(infinirtEvent_t event) {
    INFINIRT_CALL_DEVICE_API(eventSynchronize, (event));
}

__INFINI_C infiniStatus_t infinirtEventDestroy(infinirtEvent_t event) {
    INFINIRT_CALL_DEVICE_API(eventDestroy, (event));
}

__INFINI_C infiniStatus_t infinirtEventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    INFINIRT_CALL_DEVICE_API(eventElapsedTime, (ms_ptr, start, end));
}

__INFINI_C infiniStatus_t infinirtMalloc(void **p_ptr, size_t size) {
    INFINIRT_CALL_DEVICE_API(mallocDevice, (p_ptr, size));
}

__INFINI_C infiniStatus_t infinirtMallocHost(void **p_ptr, size_t size) {
    INFINIRT_CALL_DEVICE_API(mallocHost, (p_ptr, size));
}

__INFINI_C infiniStatus_t infinirtFree(void *ptr) {
    INFINIRT_CALL_DEVICE_API(freeDevice, (ptr));
}

__INFINI_C infiniStatus_t infinirtFreeHost(void *ptr) {
    INFINIRT_CALL_DEVICE_API(freeHost, (ptr));
}

__INFINI_C infiniStatus_t infinirtMemcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    INFINIRT_CALL_DEVICE_API(memcpy, (dst, src, size, kind));
}

__INFINI_C infiniStatus_t infinirtMemcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(memcpyAsync, (dst, src, size, kind, stream));
}

__INFINI_C infiniStatus_t infinirtMallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(mallocAsync, (p_ptr, size, stream));
}

__INFINI_C infiniStatus_t infinirtFreeAsync(void *ptr, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(freeAsync, (ptr, stream));
}

__INFINI_C infiniStatus_t infinirtStreamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    INFINIRT_CALL_DEVICE_API(streamBeginCapture, (stream, mode));
}

__INFINI_C infiniStatus_t infinirtStreamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) {
    INFINIRT_CALL_DEVICE_API(streamEndCapture, (stream, graph_ptr));
}

__INFINI_C infiniStatus_t infinirtGraphDestroy(infinirtGraph_t graph) {
    INFINIRT_CALL_DEVICE_API(graphDestroy, (graph));
}

__INFINI_C infiniStatus_t infinirtGraphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size) {
    INFINIRT_CALL_DEVICE_API(graphInstantiate, (graph_exec_ptr, graph, node_ptr, log_buffer, buffer_size));
}

__INFINI_C infiniStatus_t infinirtGraphExecDestroy(infinirtGraphExec_t graph_exec) {
    INFINIRT_CALL_DEVICE_API(graphExecDestroy, (graph_exec));
}

__INFINI_C infiniStatus_t infinirtGraphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    INFINIRT_CALL_DEVICE_API(graphLuanch, (graph_exec, stream));
}
