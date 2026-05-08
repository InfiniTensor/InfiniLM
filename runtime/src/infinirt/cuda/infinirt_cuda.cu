#include "../../utils.h"
#include "infinirt_cuda.cuh"
#include <cuda_runtime.h>

#define CHECK_CUDART(RT_API) CHECK_INTERNAL(RT_API, cudaSuccess)

#define RUN_CUDART(RT_API)                           \
    do {                                             \
        auto api_result_ = (RT_API);                 \
        if (api_result_ != (cudaSuccess)) {          \
            { return INFINI_STATUS_INTERNAL_ERROR; } \
        }                                            \
    } while (0)

// 根据宏定义选择命名空间并实现
#if defined(ENABLE_NVIDIA_API)
namespace infinirt::cuda {
#elif defined(ENABLE_ILUVATAR_API)
namespace infinirt::iluvatar {
#elif defined(ENABLE_QY_API)
namespace infinirt::qy {
#elif defined(ENABLE_HYGON_API)
namespace infinirt::hygon {
#elif defined(ENABLE_ALI_API)
namespace infinirt::ali {
#else
namespace infinirt::cuda { // 默认回退
#endif

infiniStatus_t getDeviceCount(int *count) {
    CHECK_CUDART(cudaGetDeviceCount(count));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t setDevice(int device_id) {
    CHECK_CUDART(cudaSetDevice(device_id));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t deviceSynchronize() {
    CHECK_CUDART(cudaDeviceSynchronize());
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamCreate(infinirtStream_t *stream_ptr) {
    cudaStream_t stream;
    CHECK_CUDART(cudaStreamCreate(&stream));
    *stream_ptr = stream;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamDestroy(infinirtStream_t stream) {
    RUN_CUDART(cudaStreamDestroy((cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamSynchronize(infinirtStream_t stream) {
    CHECK_CUDART(cudaStreamSynchronize((cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamWaitEvent(infinirtStream_t stream, infinirtEvent_t event) {
#ifdef ENABLE_ILUVATAR_API
    return INFINI_STATUS_NOT_IMPLEMENTED;
#else
    CHECK_CUDART(cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
#endif
}

infiniStatus_t eventCreate(infinirtEvent_t *event_ptr) {
    cudaEvent_t event;
    CHECK_CUDART(cudaEventCreate(&event));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventCreateWithFlags(infinirtEvent_t *event_ptr, uint32_t flags) {
    cudaEvent_t event;
    unsigned int cuda_flags = cudaEventDefault;

    // Convert infinirt flags to CUDA flags
    if (flags & INFINIRT_EVENT_DISABLE_TIMING) {
        cuda_flags |= cudaEventDisableTiming;
    }
    if (flags & INFINIRT_EVENT_BLOCKING_SYNC) {
        cuda_flags |= cudaEventBlockingSync;
    }

    CHECK_CUDART(cudaEventCreateWithFlags(&event, cuda_flags));
    *event_ptr = event;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventRecord(infinirtEvent_t event, infinirtStream_t stream) {
    CHECK_CUDART(cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventQuery(infinirtEvent_t event, infinirtEventStatus_t *status_ptr) {
    auto status = cudaEventQuery((cudaEvent_t)event);
    if (status == cudaSuccess) {
        *status_ptr = INFINIRT_EVENT_COMPLETE;
    } else if (status == cudaErrorNotReady) {
        *status_ptr = INFINIRT_EVENT_NOT_READY;
    } else {
        CHECK_CUDART(status);
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventSynchronize(infinirtEvent_t event) {
    CHECK_CUDART(cudaEventSynchronize((cudaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventDestroy(infinirtEvent_t event) {
    RUN_CUDART(cudaEventDestroy((cudaEvent_t)event));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t eventElapsedTime(float *ms_ptr, infinirtEvent_t start, infinirtEvent_t end) {
    CHECK_CUDART(cudaEventElapsedTime(ms_ptr, (cudaEvent_t)start, (cudaEvent_t)end));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocDevice(void **p_ptr, size_t size) {
    CHECK_CUDART(cudaMalloc(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocHost(void **p_ptr, size_t size) {
    CHECK_CUDART(cudaMallocHost(p_ptr, size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeDevice(void *ptr) {
    RUN_CUDART(cudaFree(ptr));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeHost(void *ptr) {
    RUN_CUDART(cudaFreeHost(ptr));
    return INFINI_STATUS_SUCCESS;
}

cudaMemcpyKind toCudaMemcpyKind(infinirtMemcpyKind_t kind) {
    switch (kind) {
    case INFINIRT_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case INFINIRT_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case INFINIRT_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    case INFINIRT_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    default:
        return cudaMemcpyDefault;
    }
}

infiniStatus_t memcpy(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind) {
    CHECK_CUDART(cudaMemcpy(dst, src, size, toCudaMemcpyKind(kind)));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t memcpyAsync(void *dst, const void *src, size_t size, infinirtMemcpyKind_t kind, infinirtStream_t stream) {
    CHECK_CUDART(cudaMemcpyAsync(dst, src, size, toCudaMemcpyKind(kind), (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t mallocAsync(void **p_ptr, size_t size, infinirtStream_t stream) {
    CHECK_CUDART(cudaMallocAsync(p_ptr, size, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t freeAsync(void *ptr, infinirtStream_t stream) {
    RUN_CUDART(cudaFreeAsync(ptr, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamBeginCapture(infinirtStream_t stream, infinirtStreamCaptureMode_t mode) {
    cudaStreamCaptureMode graph_mode;
    if (mode == INFINIRT_STREAM_CAPTURE_MODE_GLOBAL) {
        graph_mode = cudaStreamCaptureModeGlobal;
    } else if (mode == INFINIRT_STREAM_CAPTURE_MODE_THREAD_LOCAL) {
        graph_mode = cudaStreamCaptureModeThreadLocal;
    } else if (mode == INFINIRT_STREAM_CAPTURE_MODE_RELAXED) {
        graph_mode = cudaStreamCaptureModeRelaxed;
    } else {
        return INFINI_STATUS_BAD_PARAM;
    }

    CHECK_CUDART(cudaStreamBeginCapture((cudaStream_t)stream, graph_mode));

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t streamEndCapture(infinirtStream_t stream, infinirtGraph_t *graph_ptr) {
    cudaGraph_t graph;
    CHECK_CUDART(cudaStreamEndCapture((cudaStream_t)stream, &graph));
    *graph_ptr = graph;
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphDestroy(infinirtGraph_t graph) {
    RUN_CUDART(cudaGraphDestroy((cudaGraph_t)graph));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphInstantiate(
    infinirtGraphExec_t *graph_exec_ptr,
    infinirtGraph_t graph,
    infinirtGraphNode_t *node_ptr,
    char *log_buffer,
    size_t buffer_size) {
    CHECK_CUDART(cudaGraphInstantiate((cudaGraphExec_t *)graph_exec_ptr, (cudaGraph_t)graph, (cudaGraphNode_t *)node_ptr, log_buffer, buffer_size));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphExecDestroy(infinirtGraphExec_t graph_exec) {
    RUN_CUDART(cudaGraphExecDestroy((cudaGraphExec_t)graph_exec));
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t graphLuanch(infinirtGraphExec_t graph_exec, infinirtStream_t stream) {
    CHECK_CUDART(cudaGraphLaunch((cudaGraphExec_t)graph_exec, (cudaStream_t)stream));
    return INFINI_STATUS_SUCCESS;
}
}
