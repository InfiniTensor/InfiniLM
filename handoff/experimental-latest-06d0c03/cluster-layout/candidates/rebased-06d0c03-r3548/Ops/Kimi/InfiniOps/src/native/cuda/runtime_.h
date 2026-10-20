#ifndef INFINI_OPS_CUDA_RUNTIME_H_
#define INFINI_OPS_CUDA_RUNTIME_H_

#include <type_traits>

#include "runtime.h"

namespace infini::ops {

/// ## CUDA-like runtime interface enforcement via CRTP.
///
/// `CudaRuntime` extends `DeviceRuntime` for backends that mirror
/// `cuda_runtime.h`-style memory copy APIs.
template <typename Derived>
struct CudaRuntime : DeviceRuntime<Derived> {
  static constexpr bool Validate() {
    DeviceRuntime<Derived>::Validate();
    static_assert(
        std::is_invocable_v<decltype(Derived::Memcpy), void*, const void*,
                            size_t, decltype(Derived::kMemcpyHostToDevice)>,
        "`Runtime::Memcpy` must be callable with "
        "`(void*, const void*, size_t, kMemcpyHostToDevice)`.");
    return true;
  }
};

}  // namespace infini::ops

#endif
