#ifndef INFINI_OPS_TORCH_DEVICE__H_
#define INFINI_OPS_TORCH_DEVICE__H_

#include <string_view>

#include "device.h"

namespace infini::ops::detail {

template <Device::Type kDev>
struct TorchDeviceName;

template <>
struct TorchDeviceName<Device::Type::kCpu> {
  static constexpr std::string_view kValue{"cpu"};
};

template <>
struct TorchDeviceName<Device::Type::kNvidia> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kCambricon> {
  static constexpr std::string_view kValue{"mlu"};
};

template <>
struct TorchDeviceName<Device::Type::kAscend> {
  static constexpr std::string_view kValue{"npu"};
};

template <>
struct TorchDeviceName<Device::Type::kMetax> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kMoore> {
  static constexpr std::string_view kValue{"musa"};
};

template <>
struct TorchDeviceName<Device::Type::kIluvatar> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kHygon> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kMars> {
  static constexpr std::string_view kValue{"cuda"};
};

template <>
struct TorchDeviceName<Device::Type::kThead> {
  static constexpr std::string_view kValue{"cuda"};
};

}  // namespace infini::ops::detail

#endif
