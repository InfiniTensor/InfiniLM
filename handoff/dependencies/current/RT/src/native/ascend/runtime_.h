#ifndef INFINI_RT_ASCEND_RUNTIME__H_
#define INFINI_RT_ASCEND_RUNTIME__H_

#include <dlfcn.h>

#include <cassert>
#include <cstddef>
#include <cstdint>

// clang-format off
#include "acl/acl.h"
// clang-format on

#include "native/ascend/device_.h"
#include "runtime.h"

namespace infini::rt::runtime {

template <>
struct Runtime<Device::Type::kAscend>
    : GraphRuntime<Runtime<Device::Type::kAscend>> {
  using Error = aclError;

  using Stream = aclrtStream;

  using Graph = void*;

  using GraphExec = void*;

  using Event = void*;

  using StreamCaptureMode = int;

  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

  static constexpr Error kSuccess = ACL_SUCCESS;

  static Error SetDevice(int device) {
    const auto status = EnsureInitialized();
    return status == kSuccess ? aclrtSetDevice(device) : status;
  }

  static constexpr auto GetDevice = aclrtGetDevice;

  static auto GetDeviceCount(int* count) {
    assert(count != nullptr);
    const auto initialized = EnsureInitialized();
    if (initialized != kSuccess) return initialized;
    std::uint32_t device_count = 0;
    auto status = aclrtGetDeviceCount(&device_count);
    *count = static_cast<int>(device_count);
    return status;
  }

  static constexpr auto DeviceSynchronize = aclrtSynchronizeDevice;

  static constexpr auto Malloc = [](void** ptr, std::size_t size) {
    return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  };

  static Error MallocHost(void**, std::size_t) { return Unsupported(); }

  static Error MallocAsync(void**, std::size_t, Stream) {
    return Unsupported();
  }

  static constexpr auto Free = aclrtFree;

  static Error FreeHost(void*) { return Unsupported(); }

  static Error FreeAsync(void*, Stream) { return Unsupported(); }

  static Error MemGetInfo(std::size_t*, std::size_t*) { return Unsupported(); }

  static constexpr auto Memcpy = [](void* dst, const void* src,
                                    std::size_t count, aclrtMemcpyKind kind) {
    return aclrtMemcpy(dst, count, src, count, kind);
  };

  static constexpr auto MemcpyAsync = [](void* dst, const void* src,
                                         std::size_t count,
                                         aclrtMemcpyKind kind, Stream stream) {
    return aclrtMemcpyAsync(dst, count, src, count, kind, stream);
  };

  static constexpr auto kMemcpyHostToHost = ACL_MEMCPY_HOST_TO_HOST;

  static constexpr auto kMemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;

  static constexpr auto kMemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;

  static constexpr auto kMemcpyDeviceToDevice = ACL_MEMCPY_DEVICE_TO_DEVICE;

  static constexpr auto Memset = [](void* ptr, int value, std::size_t count) {
    return aclrtMemset(ptr, count, value, count);
  };

  static constexpr auto MemsetAsync = [](void* ptr, int value,
                                         std::size_t count, Stream stream) {
    return aclrtMemsetAsync(ptr, count, value, count, stream);
  };

  static constexpr auto StreamCreate = aclrtCreateStream;

  static constexpr auto StreamDestroy = aclrtDestroyStream;

  static constexpr auto StreamSynchronize = aclrtSynchronizeStream;

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

  static constexpr StreamCaptureMode kStreamCaptureModeGlobal = 0;

  static constexpr StreamCaptureMode kStreamCaptureModeThreadLocal = 1;

  static constexpr StreamCaptureMode kStreamCaptureModeRelaxed = 2;

  static Error StreamBeginCapture(Stream stream, StreamCaptureMode mode) {
    const auto& api = GraphApi();
    if (!api.Available()) {
      return UnsupportedGraphApi();
    }
    return api.CaptureBegin(stream, mode);
  }

  static Error StreamEndCapture(Stream stream, Graph* graph) {
    assert(graph != nullptr);
    const auto& api = GraphApi();
    if (!api.Available()) {
      return UnsupportedGraphApi();
    }
    int capture_status = 0;
    void* capturing_model_ri = nullptr;
    const auto info_status =
        api.CaptureGetInfo(stream, &capture_status, &capturing_model_ri);
    if (info_status != ACL_SUCCESS) {
      return info_status;
    }
    void* model_ri = nullptr;
    const auto status = api.CaptureEnd(stream, &model_ri);
    if (status != ACL_SUCCESS) {
      return status;
    }
    *graph = model_ri;
    return ACL_SUCCESS;
  }

  static Error GraphDestroy(Graph graph) {
    const auto& api = GraphApi();
    if (api.Destroy == nullptr || graph == nullptr) {
      return ACL_SUCCESS;
    }
    return api.Destroy(graph);
  }

  static Error GraphInstantiate(GraphExec* graph_exec, Graph graph) {
    assert(graph_exec != nullptr);
    *graph_exec = graph;
    return ACL_SUCCESS;
  }

  static Error GraphExecDestroy(GraphExec) { return ACL_SUCCESS; }

  static Error GraphLaunch(GraphExec graph_exec, Stream stream) {
    const auto& api = GraphApi();
    if (!api.Available()) {
      return UnsupportedGraphApi();
    }
    return api.ExecuteAsync(graph_exec, stream);
  }

 private:
  static Error EnsureInitialized() {
    // Cache one live epoch; external owners must not finalize while RT is live.
    // No static destructor finalization or post-finalize reinitialization.
    static const Error status = [] {
      const auto result = aclInit(nullptr);
      return result == ACL_ERROR_REPEAT_INITIALIZE ? kSuccess : result;
    }();
    return status;
  }

  struct RiApi {
    using CaptureBeginFn = Error (*)(Stream, int);
    using CaptureGetInfoFn = Error (*)(Stream, int*, void**);
    using CaptureEndFn = Error (*)(Stream, void**);
    using RiExecuteAsyncFn = Error (*)(void*, Stream);
    using RiDestroyFn = Error (*)(void*);

    CaptureBeginFn CaptureBegin = nullptr;
    CaptureGetInfoFn CaptureGetInfo = nullptr;
    CaptureEndFn CaptureEnd = nullptr;
    RiExecuteAsyncFn ExecuteAsync = nullptr;
    RiDestroyFn Destroy = nullptr;

    bool Available() const {
      return CaptureBegin != nullptr && CaptureGetInfo != nullptr &&
             CaptureEnd != nullptr && ExecuteAsync != nullptr &&
             Destroy != nullptr;
    }
  };

  static const RiApi& GraphApi() {
    static const RiApi api = [] {
      RiApi loaded{};
      // Some CANN releases do not expose the aclmdlRI* graph symbols in
      // headers or shared objects. Probe them at runtime so non-RI
      // environments can still build and report graph capture as unsupported.
      auto load_symbols = [](void* lib, RiApi* api) {
        api->CaptureBegin = reinterpret_cast<RiApi::CaptureBeginFn>(
            dlsym(lib, "aclmdlRICaptureBegin"));
        api->CaptureGetInfo = reinterpret_cast<RiApi::CaptureGetInfoFn>(
            dlsym(lib, "aclmdlRICaptureGetInfo"));
        api->CaptureEnd = reinterpret_cast<RiApi::CaptureEndFn>(
            dlsym(lib, "aclmdlRICaptureEnd"));
        api->ExecuteAsync = reinterpret_cast<RiApi::RiExecuteAsyncFn>(
            dlsym(lib, "aclmdlRIExecuteAsync"));
        api->Destroy =
            reinterpret_cast<RiApi::RiDestroyFn>(dlsym(lib, "aclmdlRIDestroy"));
      };

      load_symbols(RTLD_DEFAULT, &loaded);
      if (loaded.Available()) {
        return loaded;
      }

      void* lib = dlopen("libascendcl.so", RTLD_NOW | RTLD_NOLOAD);
      if (lib == nullptr) {
        lib = dlopen("libascendcl.so", RTLD_NOW);
      }
      if (lib == nullptr) {
        return loaded;
      }

      load_symbols(lib, &loaded);
      return loaded;
    }();
    return api;
  }

  static Error UnsupportedGraphApi() { return static_cast<Error>(1); }

  static Error Unsupported() { return static_cast<Error>(1); }
};

static_assert(Runtime<Device::Type::kAscend>::Validate());

}  // namespace infini::rt::runtime

#endif
