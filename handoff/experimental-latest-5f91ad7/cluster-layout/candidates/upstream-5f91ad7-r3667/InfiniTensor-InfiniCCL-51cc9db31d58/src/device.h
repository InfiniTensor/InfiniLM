#ifndef INFINI_CCL_DEVICE_H_
#define INFINI_CCL_DEVICE_H_

#include <cstdint>
#include <string>

#include "constexpr_map.h"
#include "traits.h"

namespace infini::ccl {

class Device {
 public:
  enum class Type {
    kCpu = 0,
    kNvidia = 1,
    kCambricon = 2,
    kAscend = 3,
    kMetax = 4,
    kMoore = 5,
    kIluvatar = 6,
    kKunlun = 7,
    kHygon = 8,
    kQy = 9,
    kThead = 10,
    kCount
  };

  Device() = default;

  Device(const Type &type, const int &index = 0) : type_{type}, index_{index} {}

  static const Type TypeFromString(const std::string &name) {
    return kDescToDevice.at(name);
  }

  static const std::string_view StringFromType(const Type &type) {
    return kDeviceToDesc.at(type);
  }

  const Type &type() const { return type_; }

  const int &index() const { return index_; }

  std::string ToString() const {
    return std::string{StringFromType(type_)} + ":" + std::to_string(index_);
  }

  bool operator==(const Device &other) const {
    return type_ == other.type_ && index_ == other.index_;
  }

  bool operator!=(const Device &other) const { return !(*this == other); }

 private:
  Type type_{Type::kCpu};

  static constexpr ConstexprMap<Device::Type, std::string_view,
                                static_cast<std::size_t>(Device::Type::kCount)>
      kDeviceToDesc{{{
          {Type::kCpu, "cpu"},
          {Type::kNvidia, "nvidia"},
          {Type::kCambricon, "cambricon"},
          {Type::kAscend, "ascend"},
          {Type::kMetax, "metax"},
          {Type::kMoore, "moore"},
          {Type::kIluvatar, "iluvatar"},
          {Type::kKunlun, "kunlun"},
          {Type::kHygon, "hygon"},
          {Type::kQy, "qy"},
          {Type::kThead, "thead"},
      }}};

  static constexpr ConstexprMap<std::string_view, Device::Type,
                                static_cast<std::size_t>(Device::Type::kCount)>
      kDescToDevice{{{
          {"cpu", Type::kCpu},
          {"nvidia", Type::kNvidia},
          {"cambricon", Type::kCambricon},
          {"ascend", Type::kAscend},
          {"metax", Type::kMetax},
          {"moore", Type::kMoore},
          {"iluvatar", Type::kIluvatar},
          {"kunlun", Type::kKunlun},
          {"hygon", Type::kHygon},
          {"qy", Type::kQy},
          {"thead", Type::kThead},
      }}};

  int index_{0};
};

// Primary template: Devices are disabled by default. Platform-specific
// headers (e.g. `cpu/device_.h`) specialize this to `std::true_type`.
template <Device::Type>
struct DeviceEnabled : std::false_type {};

// Defines the common categories of devices using List.
using AllDeviceTypes =
    List<Device::Type::kCpu, Device::Type::kNvidia, Device::Type::kCambricon,
         Device::Type::kAscend, Device::Type::kMetax, Device::Type::kMoore,
         Device::Type::kIluvatar, Device::Type::kKunlun, Device::Type::kHygon,
         Device::Type::kThead, Device::Type::kQy>;

// Deferred computation of active devices. The `Filter` and `FilterList`
// evaluation are nested inside a class template so that `DeviceEnabled`
// specializations from platform `device_.h` headers are visible at
// instantiation time. Use with a dependent type parameter
// (e.g. `ActiveDevices<Key>`) to ensure deferred instantiation.
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

/**
 * @brief Priority trait for device selection.
 */
template <Device::Type device_type>
struct DevicePriority {
  static constexpr int value = 0;
};

template <>
struct DevicePriority<Device::Type::kCpu> {
  static constexpr int value = 1;
};

template <>
struct DevicePriority<Device::Type::kNvidia> {
  static constexpr int value = 5;
};

template <>
struct DevicePriority<Device::Type::kIluvatar> {
  static constexpr int value = 5;
};

template <>
struct DevicePriority<Device::Type::kMetax> {
  static constexpr int value = 5;
};

template <>
struct DevicePriority<Device::Type::kMoore> {
  static constexpr int value = 5;
};

template <>
struct DevicePriority<Device::Type::kCambricon> {
  static constexpr int value = 5;
};

template <>
struct DevicePriority<Device::Type::kHygon> {
  static constexpr int value = 5;
};

template <>
struct DevicePriority<Device::Type::kThead> {
  static constexpr int value = 5;
};

enum class MemorySpace : std::uint8_t { kHost = 0, kDevice = 1, kUnknown };

template <Device::Type kDev>
MemorySpace GetMemorySpace(const void *ptr);

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICE_H_
