#ifndef INFINI_CCL_DEVICES_METAX_DEVICE_H_
#define INFINI_CCL_DEVICES_METAX_DEVICE_H_

// clang-format off
#include <mcr/mc_runtime.h>
// clang-format on

#include "checks.h"
#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kMetax> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kMetax>(const void* ptr) {
  if (!ptr) {
    return MemorySpace::kHost;
  }

  mcPointerAttribute_t attr;
  INFINI_CHECK_MACA(mcPointerGetAttributes(&attr, ptr));

  if (attr.type == mcMemoryTypeDevice || attr.type == mcMemoryTypeManaged ||
      attr.type == mcMemoryTypeArray) {
    return MemorySpace::kDevice;
  }

  return MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_METAX_DEVICE_H_
