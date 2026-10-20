#ifndef INFINI_RT_METAX_RUNTIME__H_
#define INFINI_RT_METAX_RUNTIME__H_

#include <mcr/mc_runtime.h>

#include <utility>

#include "native/cuda/metax/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kMetax>
    : GraphRuntime<Runtime<Device::Type::kMetax>,
                   CudaRuntime<Runtime<Device::Type::kMetax>>> {
  using Error = mcError_t;

  using Stream = mcStream_t;

  using Graph = mcGraph_t;

  using GraphExec = mcGraphExec_t;

  using Event = mcEvent_t;

  using StreamCaptureMode = mcStreamCaptureMode;

  static constexpr Device::Type kDeviceType = Device::Type::kMetax;

  static constexpr Error kSuccess = mcSuccess;

  static constexpr auto SetDevice = mcSetDevice;

  static constexpr auto GetDevice = mcGetDevice;

  static constexpr auto GetDeviceCount = mcGetDeviceCount;

  static constexpr auto DeviceSynchronize = mcDeviceSynchronize;

  static constexpr auto Malloc = [](auto&&... args) {
    return mcMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocHost = [](auto&&... args) {
    return mcMallocHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocAsync = [](auto&&... args) {
    return mcMallocAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto&&... args) {
    return mcFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeHost = [](auto&&... args) {
    return mcFreeHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeAsync = [](auto&&... args) {
    return mcFreeAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemGetInfo = [](auto&&... args) {
    return mcMemGetInfo(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return mcMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return mcMemcpyAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kMemcpyHostToHost = mcMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = mcMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = mcMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = mcMemcpyDeviceToDevice;

  static constexpr auto Memset = mcMemset;

  static constexpr auto MemsetAsync = [](auto&&... args) {
    return mcMemsetAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamCreate = [](auto&&... args) {
    return mcStreamCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamDestroy = [](auto&&... args) {
    return mcStreamDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = [](auto&&... args) {
    return mcStreamSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamWaitEvent = [](auto&&... args) {
    return mcStreamWaitEvent(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreate = [](auto&&... args) {
    return mcEventCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreateWithFlags = [](auto&&... args) {
    return mcEventCreateWithFlags(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventRecord = [](auto&&... args) {
    return mcEventRecord(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventQuery = [](auto&&... args) {
    return mcEventQuery(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventSynchronize = [](auto&&... args) {
    return mcEventSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventDestroy = [](auto&&... args) {
    return mcEventDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventElapsedTime = [](auto&&... args) {
    return mcEventElapsedTime(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kStreamCaptureModeGlobal = mcStreamCaptureModeGlobal;

  static constexpr auto kStreamCaptureModeThreadLocal =
      mcStreamCaptureModeThreadLocal;

  static constexpr auto kStreamCaptureModeRelaxed = mcStreamCaptureModeRelaxed;

  static constexpr auto StreamBeginCapture = [](auto&&... args) {
    return mcStreamBeginCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamEndCapture = [](auto&&... args) {
    return mcStreamEndCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphDestroy = [](auto&&... args) {
    return mcGraphDestroy(std::forward<decltype(args)>(args)...);
  };

  static Error GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    return mcGraphInstantiate(graph_exec, graph, nullptr, nullptr, 0);
  }

  static constexpr auto GraphExecDestroy = [](auto&&... args) {
    return mcGraphExecDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphLaunch = [](auto&&... args) {
    return mcGraphLaunch(std::forward<decltype(args)>(args)...);
  };
};

static_assert(Runtime<Device::Type::kMetax>::Validate());

}  // namespace infini::rt::runtime

#endif
