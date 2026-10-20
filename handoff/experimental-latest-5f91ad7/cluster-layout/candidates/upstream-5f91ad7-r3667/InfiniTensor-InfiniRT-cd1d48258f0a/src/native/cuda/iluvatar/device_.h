#ifndef INFINI_RT_ILUVATAR_DEVICE__H_
#define INFINI_RT_ILUVATAR_DEVICE__H_

#include "device.h"

namespace infini::rt {

template <>
struct DeviceEnabled<Device::Type::kIluvatar> : std::true_type {};

}  // namespace infini::rt

#endif
