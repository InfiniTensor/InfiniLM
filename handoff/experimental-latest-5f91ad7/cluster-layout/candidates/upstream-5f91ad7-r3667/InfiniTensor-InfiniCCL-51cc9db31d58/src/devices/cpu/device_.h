#ifndef INFINI_CCL_DEVICES_CPU_DEVICE_H_
#define INFINI_CCL_DEVICES_CPU_DEVICE_H_

#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kCpu> : std::true_type {};

template <>
MemorySpace GetMemorySpace<Device::Type::kCpu>(const void* /*ptr*/) {
  return MemorySpace::kHost;
}

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_CPU_DEVICE_H_
