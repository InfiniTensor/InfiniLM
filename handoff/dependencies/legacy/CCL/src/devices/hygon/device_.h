#ifndef INFINI_CCL_DEVICES_HYGON_DEVICE_H_
#define INFINI_CCL_DEVICES_HYGON_DEVICE_H_

// clang-format off
#include <hip/hip_runtime.h>
// clang-format on

#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kHygon> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kHygon>(const void *ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  hipPointerAttribute_t attr;
  hipError_t status = hipPointerGetAttributes(&attr, ptr);
  if (status != hipSuccess) {
    (void)hipGetLastError();
    return MemorySpace::kHost;
  }

  return attr.type == hipMemoryTypeDevice || attr.type == hipMemoryTypeArray ||
                 attr.type == hipMemoryTypeManaged
             ? MemorySpace::kDevice
             : MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_HYGON_DEVICE_H_
