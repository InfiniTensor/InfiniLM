#ifndef INFINI_RT_MARS_RUNTIME__H_
#define INFINI_RT_MARS_RUNTIME__H_

#include <hcr/hc_runtime.h>

#include <utility>

#include "native/cuda/mars/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kMars>
    : GraphRuntime<Runtime<Device::Type::kMars>,
                   CudaRuntime<Runtime<Device::Type::kMars>>> {
  using Error = hcError_t;

  using Stream = hcStream_t;

  using Graph = hcGraph_t;

  using GraphExec = hcGraphExec_t;

  using Event = hcEvent_t;

  using StreamCaptureMode = hcStreamCaptureMode;

  static constexpr Device::Type kDeviceType = Device::Type::kMars;

  static constexpr Error kSuccess = hcSuccess;

  static constexpr auto SetDevice = hcSetDevice;

  static constexpr auto GetDevice = hcGetDevice;

  static constexpr auto GetDeviceCount = hcGetDeviceCount;

  static constexpr auto DeviceSynchronize = hcDeviceSynchronize;

  static constexpr auto Malloc = [](auto&&... args) {
    return hcMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocHost = [](auto&&... args) {
    return hcMallocHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocAsync = [](auto&&... args) {
    return hcMallocAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return hcFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeHost = [](auto&&... args) {
    return hcFreeHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeAsync = [](auto&&... args) {
    return hcFreeAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemGetInfo = [](auto&&... args) {
    return hcMemGetInfo(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return hcMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return hcMemcpyAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kMemcpyHostToHost = hcMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = hcMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = hcMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = hcMemcpyDeviceToDevice;

  static constexpr auto Memset = hcMemset;

  static constexpr auto MemsetAsync = [](auto&&... args) {
    return hcMemsetAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamCreate = [](auto&&... args) {
    return hcStreamCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamDestroy = [](auto&&... args) {
    return hcStreamDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = [](auto&&... args) {
    return hcStreamSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamWaitEvent = [](auto&&... args) {
    return hcStreamWaitEvent(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreate = [](auto&&... args) {
    return hcEventCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreateWithFlags = [](auto&&... args) {
    return hcEventCreateWithFlags(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventRecord = [](auto&&... args) {
    return hcEventRecord(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventQuery = [](auto&&... args) {
    return hcEventQuery(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventSynchronize = [](auto&&... args) {
    return hcEventSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventDestroy = [](auto&&... args) {
    return hcEventDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventElapsedTime = [](auto&&... args) {
    return hcEventElapsedTime(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kStreamCaptureModeGlobal = hcStreamCaptureModeGlobal;

  static constexpr auto kStreamCaptureModeThreadLocal =
      hcStreamCaptureModeThreadLocal;

  static constexpr auto kStreamCaptureModeRelaxed = hcStreamCaptureModeRelaxed;

  static constexpr auto StreamBeginCapture = [](auto&&... args) {
    return hcStreamBeginCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamEndCapture = [](auto&&... args) {
    return hcStreamEndCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphDestroy = [](auto&&... args) {
    return hcGraphDestroy(std::forward<decltype(args)>(args)...);
  };

  static Error GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    return hcGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
  }

  static constexpr auto GraphExecDestroy = [](auto&&... args) {
    return hcGraphExecDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphLaunch = [](auto&&... args) {
    return hcGraphLaunch(std::forward<decltype(args)>(args)...);
  };
};

static_assert(Runtime<Device::Type::kMars>::Validate());

}  // namespace infini::rt::runtime

#endif
