#ifndef INFINI_RT_CPU_DEVICE__H_
#define INFINI_RT_CPU_DEVICE__H_

#include "device.h"

namespace infini::rt {

template <>
struct DeviceEnabled<Device::Type::kCpu> : std::true_type {};

}  // namespace infini::rt

#endif
