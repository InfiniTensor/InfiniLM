#ifndef INFINI_RT_CAMBRICON_RUNTIME__H_
#define INFINI_RT_CAMBRICON_RUNTIME__H_

#include <cnrt.h>

#include <cassert>
#include <cstddef>

#include "native/cambricon/device_.h"
#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kCambricon>
    : GraphRuntime<Runtime<Device::Type::kCambricon>> {
  using Error = cnrtRet_t;

  using Stream = cnrtQueue_t;

  // CNRT exposes graph capture and replay as TaskTopo objects.
  using Graph = cnrtTaskTopo_t;

  using GraphExec = cnrtTaskTopoEntity_t;

  using Event = void*;

  using StreamCaptureMode = cnrtQueueCaptureMode_t;

  static constexpr Device::Type kDeviceType = Device::Type::kCambricon;

#ifdef CNRT_RET_SUCCESS
  static constexpr Error kSuccess = CNRT_RET_SUCCESS;
#else
  static constexpr Error kSuccess = cnrtSuccess;
#endif

  static constexpr auto SetDevice = cnrtSetDevice;

  static constexpr auto GetDevice = cnrtGetDevice;

  static auto GetDeviceCount(int* count) {
    assert(count != nullptr);
    unsigned int device_count = 0;
    auto status = cnrtGetDeviceCount(&device_count);
    *count = static_cast<int>(device_count);
    return status;
  }

  static constexpr auto DeviceSynchronize = cnrtSyncDevice;

  static constexpr auto Malloc = cnrtMalloc;

  static Error MallocHost(void**, std::size_t) { return Unsupported(); }

  static Error MallocAsync(void**, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto Free = cnrtFree;

  static Error FreeHost(void*) { return Unsupported(); }

  static Error FreeAsync(void*, Stream) { return Unsupported(); }

  static Error MemGetInfo(std::size_t*, std::size_t*) { return Unsupported(); }

  static constexpr auto Memcpy = [](void* dst, const void* src,
                                    std::size_t size, auto kind) {
    return cnrtMemcpy(dst, const_cast<void*>(src), size, kind);
  };

  static constexpr auto MemcpyAsync = [](void* dst, const void* src,
                                         std::size_t size, auto kind,
                                         Stream stream) {
    return cnrtMemcpyAsync_V2(dst, const_cast<void*>(src), size, stream, kind);
  };

  static constexpr auto kMemcpyHostToHost = cnrtMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = cnrtMemcpyHostToDev;

  static constexpr auto kMemcpyDeviceToHost = cnrtMemcpyDevToHost;

  static constexpr auto kMemcpyDeviceToDevice = cnrtMemcpyDevToDev;

  static constexpr auto Memset = cnrtMemset;

  // InfiniCore emits zero-fill work on the captured queue.
  static constexpr auto MemsetAsync = cnrtMemsetAsync;

  static constexpr auto StreamCreate = cnrtQueueCreate;

  static constexpr auto StreamDestroy = cnrtQueueDestroy;

  static constexpr auto StreamSynchronize = cnrtQueueSync;

  static Error StreamWaitEvent(Stream, Event, unsigned int) {
    return Unsupported();
  }

  static Error EventCreate(Event*) { return Unsupported(); }

  static Error EventCreateWithFlags(Event*, unsigned int) {
    return Unsupported();
  }

  static Error EventRecord(Event, Stream) { return Unsupported(); }

  static Error EventQuery(Event) { return Unsupported(); }

  static Error EventSynchronize(Event) { return Unsupported(); }

  static Error EventDestroy(Event) { return Unsupported(); }

  static Error EventElapsedTime(float*, Event, Event) { return Unsupported(); }

  static constexpr auto kStreamCaptureModeGlobal = cnrtQueueCaptureModeGlobal;

  static constexpr auto kStreamCaptureModeThreadLocal =
      cnrtQueueCaptureModeThreadLocal;

  static constexpr auto kStreamCaptureModeRelaxed = cnrtQueueCaptureModeRelaxed;

  static constexpr auto StreamBeginCapture = cnrtQueueBeginCapture;

  static Error StreamEndCapture(Stream stream, Graph* graph) {
    assert(graph != nullptr);
    return cnrtQueueEndCapture(stream, graph);
  }

  static Error GraphDestroy(Graph graph) {
    return graph == nullptr ? kSuccess : cnrtTaskTopoDestroy(graph);
  }

  static Error GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    assert(graph_exec != nullptr);
    return cnrtTaskTopoInstantiate(graph_exec, graph, nullptr, nullptr, 0);
  }

  static Error GraphExecDestroy(GraphExec graph_exec) {
    return graph_exec == nullptr ? kSuccess
                                 : cnrtTaskTopoEntityDestroy(graph_exec);
  }

  static constexpr auto GraphLaunch = cnrtTaskTopoEntityInvoke;

 private:
  static Error Unsupported() { return static_cast<Error>(1); }
};

static_assert(Runtime<Device::Type::kCambricon>::Validate());

}  // namespace infini::rt::runtime

#endif
