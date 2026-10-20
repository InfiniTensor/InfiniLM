#ifndef INFINI_RT_CUDA_DRIVER__H_
#define INFINI_RT_CUDA_DRIVER__H_

#include <type_traits>

#include "driver.h"

namespace infini::rt::driver {

template <typename Derived>
struct CudaDriver : DeviceDriver<Derived> {
  static constexpr bool Validate() {
    DeviceDriver<Derived>::Validate();
    static_assert(sizeof(typename Derived::Function) > 0,
                  "`Driver` must define a `Function` type alias.");
    static_assert(sizeof(typename Derived::Module) > 0,
                  "`Driver` must define a `Module` type alias.");
    static_assert(sizeof(typename Derived::Stream) > 0,
                  "`Driver` must define a `Stream` type alias.");
    static_assert(std::is_invocable_v<decltype(Derived::ModuleLoadData),
                                      typename Derived::Module*, const void*>,
                  "`Driver::ModuleLoadData` must be callable with `(Module*, "
                  "const void*)`.");
    static_assert(std::is_invocable_v<decltype(Derived::ModuleGetFunction),
                                      typename Derived::Function*,
                                      typename Derived::Module, const char*>,
                  "`Driver::ModuleGetFunction` must be callable with "
                  "`(Function*, Module, const char*)`.");
    static_assert(std::is_invocable_v<decltype(Derived::ModuleUnload),
                                      typename Derived::Module>,
                  "`Driver::ModuleUnload` must be callable with `(Module)`.");
    static_assert(
        std::is_invocable_v<decltype(Derived::LaunchKernel),
                            typename Derived::Function, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            unsigned int, unsigned int, unsigned int,
                            typename Derived::Stream, void**, void**>,
        "`Driver::LaunchKernel` must be callable with "
        "`(Function, unsigned int, unsigned int, unsigned int, unsigned int, "
        "unsigned int, unsigned int, unsigned int, Stream, void**, void**)`.");
    return true;
  }
};

}  // namespace infini::rt::driver

#endif
