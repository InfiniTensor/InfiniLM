#ifndef INFINI_RT_MOORE_RUNTIME__H_
#define INFINI_RT_MOORE_RUNTIME__H_

#include <musa_runtime.h>

#include <cstddef>
#include <utility>

#include "native/cuda/moore/device_.h"
#include "native/cuda/runtime_.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kMoore>
    : GraphRuntime<Runtime<Device::Type::kMoore>,
                   CudaRuntime<Runtime<Device::Type::kMoore>>> {
  using Error = musaError_t;

  using Stream = musaStream_t;

  using Graph = musaGraph_t;

  using GraphExec = musaGraphExec_t;

  using Event = musaEvent_t;

  using StreamCaptureMode = musaStreamCaptureMode;

  static constexpr Device::Type kDeviceType = Device::Type::kMoore;

  static constexpr Error kSuccess = musaSuccess;

  static constexpr auto SetDevice = musaSetDevice;

  static constexpr auto GetDevice = [](auto&&... args) {
    return musaGetDevice(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GetDeviceCount = [](auto&&... args) {
    return musaGetDeviceCount(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto DeviceSynchronize = [](auto&&... args) {
    return musaDeviceSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Malloc = [](auto&&... args) {
    return musaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocHost = [](auto&&... args) {
    return musaMallocHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MallocAsync = [](void**, std::size_t, Stream) {
    return static_cast<Error>(1);
  };

  static constexpr auto Free = [](auto&&... args) {
    return musaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeHost = [](auto&&... args) {
    return musaFreeHost(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto FreeAsync = [](void*, Stream) {
    return static_cast<Error>(1);
  };

  static constexpr auto MemGetInfo = [](auto&&... args) {
    return musaMemGetInfo(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto&&... args) {
    return musaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyAsync = [](auto&&... args) {
    return musaMemcpyAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kMemcpyHostToHost = musaMemcpyHostToHost;

  static constexpr auto kMemcpyHostToDevice = musaMemcpyHostToDevice;

  static constexpr auto kMemcpyDeviceToHost = musaMemcpyDeviceToHost;

  static constexpr auto kMemcpyDeviceToDevice = musaMemcpyDeviceToDevice;

  static constexpr auto Memset = musaMemset;

  static constexpr auto MemsetAsync = [](auto&&... args) {
    return musaMemsetAsync(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamCreate = [](auto&&... args) {
    return musaStreamCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamDestroy = [](auto&&... args) {
    return musaStreamDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = [](auto&&... args) {
    return musaStreamSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamWaitEvent = [](auto&&... args) {
    return musaStreamWaitEvent(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreate = [](auto&&... args) {
    return musaEventCreate(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventCreateWithFlags = [](auto&&... args) {
    return musaEventCreateWithFlags(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventRecord = [](auto&&... args) {
    return musaEventRecord(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventQuery = [](auto&&... args) {
    return musaEventQuery(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventSynchronize = [](auto&&... args) {
    return musaEventSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventDestroy = [](auto&&... args) {
    return musaEventDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto EventElapsedTime = [](auto&&... args) {
    return musaEventElapsedTime(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto kStreamCaptureModeGlobal = musaStreamCaptureModeGlobal;

  static constexpr auto kStreamCaptureModeThreadLocal =
      musaStreamCaptureModeThreadLocal;

  static constexpr auto kStreamCaptureModeRelaxed =
      musaStreamCaptureModeRelaxed;

  static constexpr auto StreamBeginCapture = [](auto&&... args) {
    return musaStreamBeginCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamEndCapture = [](auto&&... args) {
    return musaStreamEndCapture(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphDestroy = [](auto&&... args) {
    return musaGraphDestroy(std::forward<decltype(args)>(args)...);
  };

  static Error GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    return musaGraphInstantiate(graph_exec, graph, 0);
  }

  static constexpr auto GraphExecDestroy = [](auto&&... args) {
    return musaGraphExecDestroy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GraphLaunch = [](auto&&... args) {
    return musaGraphLaunch(std::forward<decltype(args)>(args)...);
  };
};

static_assert(Runtime<Device::Type::kMoore>::Validate());

}  // namespace infini::rt::runtime

#endif
