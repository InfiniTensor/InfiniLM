#ifndef INFINI_CCL_OPERATION_H_
#define INFINI_CCL_OPERATION_H_

#include <initializer_list>
#include <memory>

#include "backend.h"
#include "backend_device_map.h"
#include "device.h"
#include "dispatcher.h"
#include "return_status_impl.h"
#include "traits.h"

namespace infini::ccl {

template <typename Key, BackendType backend_type = BackendType::kCount,
          Device::Type device_type = Device::Type::kCount>
class Operation {
 public:
  template <typename... Args>
  static auto Call(Args &&...args) {
    constexpr BackendType kBestBack =
        ListGetBest<BackendPriority>(ActiveBackends<Key>{});
    constexpr Device::Type kBestDev =
        ListGetBest<DevicePriority>(ActiveDevices<Key>{});

    return Call(kBestBack, kBestDev, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto Call(BackendType backend, Device::Type device, Args &&...args) {
    return DispatchFunc<ActiveBackends<Key>, ActiveDevices<Key>>(
        {static_cast<int64_t>(backend), static_cast<int64_t>(device)},
        [&](auto resolved_list) {
          constexpr BackendType kBackend =
              static_cast<BackendType>(ListGet<0>(resolved_list));
          constexpr Device::Type kDevice =
              static_cast<Device::Type>(ListGet<1>(resolved_list));

          if constexpr (IsSupportedCombination<kBackend, kDevice>::value) {
            return Key::template Execute<kBackend, kDevice>(
                std::forward<Args>(args)...);
          } else {
            return ReturnStatus::kNotSupported;
          }
        },
        "Operation::Call");
  }

  static BackendType FindSupportedBackend(
      Device::Type device, std::initializer_list<BackendType> candidates) {
    for (BackendType candidate : candidates) {
      if (Supports(candidate, device)) {
        return candidate;
      }
    }
    return BackendType::kCount;
  }

 private:
  template <auto device, auto... backends>
  static constexpr bool SupportsBackend(BackendType backend,
                                        List<backends...>) {
    return ((backend == backends &&
             IsSupportedCombination<backends, device>::value) ||
            ...);
  }

  template <auto... devices>
  static constexpr bool Supports(BackendType backend, Device::Type device,
                                 List<devices...>) {
    return ((device == devices &&
             SupportsBackend<devices>(backend, ActiveBackends<Key>{})) ||
            ...);
  }

  static constexpr bool Supports(BackendType backend, Device::Type device) {
    return Supports(backend, device, ActiveDevices<Key>{});
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_OPERATION_H_
