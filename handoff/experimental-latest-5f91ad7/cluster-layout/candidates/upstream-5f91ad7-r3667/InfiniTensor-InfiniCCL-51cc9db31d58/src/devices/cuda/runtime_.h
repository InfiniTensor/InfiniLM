#ifndef INFINI_CCL_DEVICES_CUDA_RUNTIME_H_
#define INFINI_CCL_DEVICES_CUDA_RUNTIME_H_

#include <type_traits>

#include "runtime.h"

namespace infini::ccl {

template <typename Derived>
struct CudaRuntime : DeviceRuntime<Derived> {
  static constexpr bool Validate() {
    DeviceRuntime<Derived>::Validate();
    static_assert(
        std::is_invocable_v<decltype(Derived::Memcpy), void *, const void *,
                            size_t, decltype(Derived::MemcpyHostToDevice)>,
        "`Runtime::Memcpy` must be callable with "
        "`(void*, const void*, size_t, MemcpyHostToDevice)`.");
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CUDA_RUNTIME_H_
