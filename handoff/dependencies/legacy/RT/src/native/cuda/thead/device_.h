#ifndef INFINI_RT_THEAD_DEVICE__H_
#define INFINI_RT_THEAD_DEVICE__H_

#include "device.h"

namespace infini::rt {

template <>
struct DeviceEnabled<Device::Type::kThead> : std::true_type {};

}  // namespace infini::rt

#endif
