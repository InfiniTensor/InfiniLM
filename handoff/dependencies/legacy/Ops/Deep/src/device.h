#ifndef INFINI_OPS_DEVICE_H_
#define INFINI_OPS_DEVICE_H_

#include <infini/rt.h>

#include "common/traits.h"

namespace infini::ops {

using Device = infini::rt::Device;

template <Device::Type device_type>
using DeviceEnabled = infini::rt::DeviceEnabled<device_type>;

using AllDeviceTypes =
    List<Device::Type::kCpu, Device::Type::kNvidia, Device::Type::kCambricon,
         Device::Type::kAscend, Device::Type::kMetax, Device::Type::kMoore,
         Device::Type::kIluvatar, Device::Type::kHygon, Device::Type::kMars,
         Device::Type::kThead>;

template <typename>
struct ActiveDevicesImpl {
  struct Filter {
    template <Device::Type kDev>
    std::enable_if_t<DeviceEnabled<kDev>::value> operator()(
        ValueTag<kDev>) const {}
  };

  using type = typename FilterList<Filter, std::tuple<>, AllDeviceTypes>::type;
};

template <typename T>
using ActiveDevices = typename ActiveDevicesImpl<T>::type;

}  // namespace infini::ops

#endif
