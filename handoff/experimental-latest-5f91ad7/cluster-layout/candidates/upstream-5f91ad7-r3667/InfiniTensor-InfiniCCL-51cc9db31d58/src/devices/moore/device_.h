#ifndef INFINI_CCL_DEVICES_MOORE_DEVICE_H_
#define INFINI_CCL_DEVICES_MOORE_DEVICE_H_

#include <musa_runtime.h>

#include "checks.h"
#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kMoore> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kMoore>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  musaPointerAttributes attr;
  INFINI_CHECK_MUSA(musaPointerGetAttributes(&attr, ptr));

  if (attr.type == musaMemoryTypeDevice || attr.type == musaMemoryTypeManaged) {
    return MemorySpace::kDevice;
  }

  return MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MOORE_DEVICE_H_
