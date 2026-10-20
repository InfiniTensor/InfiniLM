#ifndef INFINI_RT_CAMBRICON_DEVICE__H_
#define INFINI_RT_CAMBRICON_DEVICE__H_

#include "device.h"

namespace infini::rt {

template <>
struct DeviceEnabled<Device::Type::kCambricon> : std::true_type {};

}  // namespace infini::rt

#endif
