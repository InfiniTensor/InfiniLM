#ifndef INFINI_RT_DRIVER_H_
#define INFINI_RT_DRIVER_H_

#include <type_traits>

#include "device.h"

namespace infini::rt::driver {

template <Device::Type device_type>
struct Driver;

template <typename Derived>
struct DriverBase {
  static constexpr bool Validate() {
    static_assert(
        std::is_same_v<std::remove_cv_t<decltype(Derived::kDeviceType)>,
                       Device::Type>,
        "`Driver` must define `static constexpr Device::Type kDeviceType`.");
    static_assert(sizeof(typename Derived::Result) > 0,
                  "`Driver` must define a `Result` type alias.");
    static_assert(std::is_same_v<std::remove_cv_t<decltype(Derived::kSuccess)>,
                                 typename Derived::Result>,
                  "`Driver` must define `static constexpr Result kSuccess`.");
    return true;
  }
};

template <typename Derived>
using DeviceDriver = DriverBase<Derived>;

}  // namespace infini::rt::driver

#endif
