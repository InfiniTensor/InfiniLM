#ifndef INFINI_CCL_DEVICES_CAMBRICON_DEVICE_H_
#define INFINI_CCL_DEVICES_CAMBRICON_DEVICE_H_

#include <cnrt.h>

#include "checks.h"
#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kCambricon> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kCambricon>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  cnrtPointerAttributes_t attr;
  INFINI_CHECK_CNRT(cnrtPointerGetAttributes(&attr, ptr));

  if (attr.type == cnrtMemTypeDevice) {
    return MemorySpace::kDevice;
  }

  return MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CAMBRICON_DEVICE_H_
