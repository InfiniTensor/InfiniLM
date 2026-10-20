#ifndef INFINI_RT_RUNTIME_H_
#define INFINI_RT_RUNTIME_H_

#include <cstddef>
#include <type_traits>

#include "device.h"

namespace infini::rt::runtime {

template <Device::Type device_type>
struct Runtime;

/// ## Interface enforcement via CRTP.
///
/// Inherit from the appropriate base to declare which interface level a
/// `runtime::Runtime` specialization implements. After the struct is fully
/// defined, call `static_assert(Runtime<...>::Validate())`. The chained
/// `Validate()` checks
/// every required member's existence and signature at compile time, analogous
/// to how `override` catches signature mismatches for virtual functions.
///
/// - `RuntimeBase`: `kDeviceType` only (e.g. CPU).
/// - `DeviceRuntime`: adds `Stream`, `Malloc`, and `Free` (e.g. Cambricon).
/// - `GraphRuntime`: adds graph capture/replay types and APIs.

/// Every Runtime must provide `static constexpr Device::Type kDeviceType`.
template <typename Derived>
struct RuntimeBase {
  static constexpr bool Validate() {
    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Derived::kDeviceType)>,
                       Device::Type>,
        "`Runtime` must define `static constexpr Device::Type kDeviceType`.");
    static_assert(sizeof(typename Derived::Error) > 0,
                  "`Runtime` must define an `Error` type alias.");
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Derived::kSuccess)>,
                                 typename Derived::Error>,
                  "`Runtime` must define `static constexpr Error kSuccess`.");
    return true;
  }
};

/// Runtimes with device memory must additionally provide `Stream`, `Malloc`,
/// `Free`, and basic stream lifecycle APIs.
template <typename Derived>
struct DeviceRuntime : RuntimeBase<Derived> {
  static constexpr bool Validate() {
    RuntimeBase<Derived>::Validate();
    static_assert(sizeof(typename Derived::Stream) > 0,
                  "`Runtime` must define a `Stream` type alias.");
    static_assert(
        std::is_invocable_v<decltype(Derived::Malloc), void**, size_t>,
        "`Runtime::Malloc` must be callable with `(void**, size_t)`.");
    static_assert(std::is_invocable_v<decltype(Derived::Free), void*>,
                  "`Runtime::Free` must be callable with `(void*)`.");
    static_assert(std::is_invocable_v<decltype(Derived::StreamCreate),
                                      typename Derived::Stream*>,
                  "`Runtime::StreamCreate` must be callable with `(Stream*)`.");
    static_assert(std::is_invocable_v<decltype(Derived::StreamDestroy),
                                      typename Derived::Stream>,
                  "`Runtime::StreamDestroy` must be callable with `(Stream)`.");
    static_assert(
        std::is_invocable_v<decltype(Derived::StreamSynchronize),
                            typename Derived::Stream>,
        "`Runtime::StreamSynchronize` must be callable with `(Stream)`.");
    return true;
  }
};

/// Runtimes with graph capture/replay must additionally provide graph types and
/// graph lifecycle APIs. The optional `Base` parameter lets platform families
/// layer graph validation on top of a more specific runtime contract, such as
/// `CudaRuntime`.
template <typename Derived, typename Base = DeviceRuntime<Derived>>
struct GraphRuntime : Base {
  static constexpr bool Validate() {
    Base::Validate();
    static_assert(sizeof(typename Derived::Graph) > 0,
                  "`Runtime` must define a `Graph` type alias.");
    static_assert(sizeof(typename Derived::GraphExec) > 0,
                  "`Runtime` must define a `GraphExec` type alias.");
    static_assert(sizeof(typename Derived::StreamCaptureMode) > 0,
                  "`Runtime` must define a `StreamCaptureMode` type alias.");
    static_assert(std::is_invocable_v<decltype(Derived::StreamBeginCapture),
                                      typename Derived::Stream,
                                      typename Derived::StreamCaptureMode>,
                  "`Runtime::StreamBeginCapture` must be callable with "
                  "`(Stream, StreamCaptureMode)`.");
    static_assert(
        std::is_invocable_v<decltype(Derived::StreamEndCapture),
                            typename Derived::Stream, typename Derived::Graph*>,
        "`Runtime::StreamEndCapture` must be callable with "
        "`(Stream, Graph*)`.");
    static_assert(std::is_invocable_v<decltype(Derived::GraphDestroy),
                                      typename Derived::Graph>,
                  "`Runtime::GraphDestroy` must be callable with `(Graph)`.");
    static_assert(std::is_invocable_v<decltype(Derived::GraphInstantiate),
                                      typename Derived::GraphExec*,
                                      typename Derived::Graph>,
                  "`Runtime::GraphInstantiate` must be callable with "
                  "`(GraphExec*, Graph)`.");
    static_assert(
        std::is_invocable_v<decltype(Derived::GraphExecDestroy),
                            typename Derived::GraphExec>,
        "`Runtime::GraphExecDestroy` must be callable with `(GraphExec)`.");
    static_assert(
        std::is_invocable_v<decltype(Derived::GraphLaunch),
                            typename Derived::GraphExec,
                            typename Derived::Stream>,
        "`Runtime::GraphLaunch` must be callable with `(GraphExec, Stream)`.");
    return true;
  }
};

}  // namespace infini::rt::runtime

#endif
