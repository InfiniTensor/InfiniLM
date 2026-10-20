#ifndef INFINI_OPS_PYBIND11_UTILS_H_
#define INFINI_OPS_PYBIND11_UTILS_H_

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <string_view>
#include <type_traits>

#include "data_type.h"
#include "host_range_profiler.h"
#include "tensor.h"
#include "torch/device_.h"

namespace py = pybind11;

namespace infini::ops {

namespace detail {

inline py::handle InternedName(const char* value) {
  // Keep one reference for the process lifetime; Python objects must not be
  // decref'd by a static destructor after interpreter finalization.
  auto* name{PyUnicode_InternFromString(value)};
  if (name == nullptr) throw py::error_already_set();
  return py::handle{name};
}

struct InternedNames {
  py::handle data_ptr{InternedName("data_ptr")};
  py::handle shape{InternedName("shape")};
  py::handle dtype{InternedName("dtype")};
  py::handle device{InternedName("device")};
  py::handle type{InternedName("type")};
  py::handle index{InternedName("index")};
  py::handle stride{InternedName("stride")};
};

inline const InternedNames& GetInternedNames() {
  // Defer Python C API calls until conversion runs with an active interpreter.
  static const InternedNames names;
  return names;
}

inline py::object CallMethodNoArgs(py::handle obj, py::handle name) {
  auto* value{PyObject_CallMethodNoArgs(obj.ptr(), name.ptr())};
  if (value == nullptr) throw py::error_already_set();
  return py::reinterpret_steal<py::object>(value);
}

template <typename Integer>
Integer IntegerFromPybind11Handle(py::handle obj) {
  static_assert(std::is_integral_v<Integer>);
  if constexpr (std::is_unsigned_v<Integer>) {
    const auto value{PyLong_AsUnsignedLongLong(obj.ptr())};
    if (value == static_cast<unsigned long long>(-1) && PyErr_Occurred()) {
      throw py::error_already_set();
    }
    if (value > std::numeric_limits<Integer>::max()) {
      PyErr_SetString(PyExc_OverflowError,
                      "integer out of range for tensor metadata");
      throw py::error_already_set();
    }
    return static_cast<Integer>(value);
  } else {
    const auto value{PyLong_AsLongLong(obj.ptr())};
    if (value == -1 && PyErr_Occurred()) throw py::error_already_set();
    if (value < std::numeric_limits<Integer>::min() ||
        value > std::numeric_limits<Integer>::max()) {
      PyErr_SetString(PyExc_OverflowError,
                      "integer out of range for tensor metadata");
      throw py::error_already_set();
    }
    return static_cast<Integer>(value);
  }
}

template <typename Vector>
Vector VectorFromPybind11Handle(py::handle obj) {
  auto* sequence_ptr{PySequence_Fast(obj.ptr(), "expected a sequence")};
  if (sequence_ptr == nullptr) throw py::error_already_set();
  auto sequence{py::reinterpret_steal<py::object>(sequence_ptr)};
  const auto size{PySequence_Fast_GET_SIZE(sequence.ptr())};

  Vector result;
  result.reserve(static_cast<std::size_t>(size));
  for (Py_ssize_t i = 0; i < size; ++i) {
    result.push_back(IntegerFromPybind11Handle<typename Vector::value_type>(
        py::handle{PySequence_Fast_GET_ITEM(sequence.ptr(), i)}));
  }
  return result;
}

template <Device::Type... kDevs>
std::unordered_map<std::string, Device::Type> BuildTorchNameMap(
    List<kDevs...>) {
  std::unordered_map<std::string, Device::Type> map;
  (map.emplace(std::string{TorchDeviceName<kDevs>::kValue}, kDevs), ...);
  return map;
}

}  // namespace detail

inline DataType DataTypeFromString(std::string_view name) {
  return kStringToDataType.at(name);
}

namespace detail {

inline DataType DataTypeFromPybind11HandleImpl(py::handle obj) {
  py::str dtype_obj{obj};
  Py_ssize_t size{0};
  const char* data{PyUnicode_AsUTF8AndSize(dtype_obj.ptr(), &size)};
  if (data == nullptr) throw py::error_already_set();
  std::string_view dtype_str{data, static_cast<std::size_t>(size)};
  const auto pos{dtype_str.find_last_of('.')};

  return DataTypeFromString(
      pos == std::string_view::npos ? dtype_str : dtype_str.substr(pos + 1));
}

}  // namespace detail

inline DataType DataTypeFromPybind11Handle(py::handle obj) {
  [[maybe_unused]] HostRangeScope host_range_tensor_conversion{
      HostRangeLayer::kTensorConversion};
  return detail::DataTypeFromPybind11HandleImpl(obj);
}

template <typename T = void>
inline Device::Type DeviceTypeFromString(std::string_view name) {
  static const auto kTorchNameToTypes{
      detail::BuildTorchNameMap(ActiveDevices<T>{})};

  auto it{
      std::find_if(kTorchNameToTypes.cbegin(), kTorchNameToTypes.cend(),
                   [name](const auto& item) { return item.first == name; })};

  if (it != kTorchNameToTypes.cend()) {
    return it->second;
  }

  std::vector<std::string> supported_names;

  for (const auto& [torch_name, device_type] : kTorchNameToTypes) {
    const auto internal_name = Device::StringFromType(device_type);

    if (name == internal_name) {
      return device_type;
    }

    supported_names.push_back(torch_name);
    supported_names.emplace_back(internal_name);
  }

  std::sort(supported_names.begin(), supported_names.end());
  supported_names.erase(
      std::unique(supported_names.begin(), supported_names.end()),
      supported_names.end());

  std::string message{"Unsupported device type `"};
  message.append(name.data(), name.size());
  message += "` for this InfiniOps build. Supported device names: ";

  for (std::size_t i = 0; i < supported_names.size(); ++i) {
    if (i != 0) {
      message += ", ";
    }
    message += supported_names[i];
  }

  message += ". Rebuild InfiniOps with the matching backend enabled.";

  throw py::value_error(message);
}

// Returns `nullopt` rather than aborting when the name does not resolve.
// Used by generated pybind bindings to query implementation indices for
// devices an op may not support, without crashing the process.
template <typename T = void>
inline std::optional<Device::Type> TryDeviceTypeFromString(
    std::string_view name) {
  static const auto kTorchNameToTypes{
      detail::BuildTorchNameMap(ActiveDevices<T>{})};

  auto it{
      std::find_if(kTorchNameToTypes.cbegin(), kTorchNameToTypes.cend(),
                   [name](const auto& item) { return item.first == name; })};

  if (it != kTorchNameToTypes.cend()) {
    return it->second;
  }

  static const std::unordered_map<std::string, Device::Type> kPlatformNames{
      {"cpu", Device::Type::kCpu},
      {"nvidia", Device::Type::kNvidia},
      {"cambricon", Device::Type::kCambricon},
      {"ascend", Device::Type::kAscend},
      {"metax", Device::Type::kMetax},
      {"mars", Device::Type::kMars},
      {"moore", Device::Type::kMoore},
      {"iluvatar", Device::Type::kIluvatar},
      {"hygon", Device::Type::kHygon},
      {"thead", Device::Type::kThead},
  };

  auto platform_it{
      std::find_if(kPlatformNames.cbegin(), kPlatformNames.cend(),
                   [name](const auto& item) { return item.first == name; })};

  if (platform_it != kPlatformNames.cend()) {
    return platform_it->second;
  }

  return std::nullopt;
}

namespace detail {

inline Device DeviceFromPybind11HandleImpl(py::handle obj) {
  const auto& names{GetInternedNames()};
  auto device_obj{py::getattr(obj, names.device)};
  auto device_type_obj{py::getattr(device_obj, names.type)};
  std::string device_type_storage;
  std::string_view device_type_str;
  if (PyUnicode_Check(device_type_obj.ptr())) {
    Py_ssize_t device_type_size{0};
    const char* device_type_data{
        PyUnicode_AsUTF8AndSize(device_type_obj.ptr(), &device_type_size)};
    if (device_type_data == nullptr) throw py::error_already_set();
    device_type_str = {device_type_data,
                       static_cast<std::size_t>(device_type_size)};
  } else {
    device_type_storage = device_type_obj.cast<std::string>();
    device_type_str = device_type_storage;
  }
  auto device_index_obj{py::getattr(device_obj, names.index)};
  auto device_index{device_index_obj.is_none() ? 0
                                               : device_index_obj.cast<int>()};

  return Device{DeviceTypeFromString(device_type_str), device_index};
}

inline Tensor TensorFromPybind11HandleImpl(py::handle obj) {
  const auto& names{GetInternedNames()};
  auto data{reinterpret_cast<void*>(
      detail::CallMethodNoArgs(obj, names.data_ptr).cast<std::uintptr_t>())};

  auto shape{detail::VectorFromPybind11Handle<typename Tensor::Shape>(
      py::getattr(obj, names.shape))};

  auto dtype{DataTypeFromPybind11HandleImpl(py::getattr(obj, names.dtype))};

  auto device{DeviceFromPybind11HandleImpl(obj)};

  auto strides{detail::VectorFromPybind11Handle<typename Tensor::Strides>(
      detail::CallMethodNoArgs(obj, names.stride))};

  return Tensor{data, std::move(shape), dtype, device, std::move(strides)};
}

}  // namespace detail

inline Device DeviceFromPybind11Handle(py::handle obj) {
  [[maybe_unused]] HostRangeScope host_range_device_conversion{
      HostRangeLayer::kDeviceConversion};
  return detail::DeviceFromPybind11HandleImpl(obj);
}

inline Tensor TensorFromPybind11Handle(py::handle obj) {
  [[maybe_unused]] HostRangeScope host_range_tensor_conversion{
      HostRangeLayer::kTensorConversion};
  return detail::TensorFromPybind11HandleImpl(obj);
}

inline std::optional<Tensor> OptionalTensorFromPybind11Handle(
    const std::optional<py::object>& obj) {
  [[maybe_unused]] HostRangeScope host_range_tensor_conversion{
      HostRangeLayer::kTensorConversion};
  if (!obj.has_value() || obj->is_none()) return std::nullopt;
  return detail::TensorFromPybind11HandleImpl(*obj);
}

inline std::vector<Tensor> VectorTensorFromPybind11Handle(
    const std::vector<py::object>& objs) {
  [[maybe_unused]] HostRangeScope host_range_tensor_conversion{
      HostRangeLayer::kTensorConversion};
  std::vector<Tensor> result;
  result.reserve(objs.size());
  for (const auto& obj : objs) {
    result.push_back(detail::TensorFromPybind11HandleImpl(obj));
  }
  return result;
}

inline std::vector<std::optional<Tensor>>
VectorOptionalTensorFromPybind11Handle(const std::vector<py::object>& objs) {
  [[maybe_unused]] HostRangeScope host_range_tensor_conversion{
      HostRangeLayer::kTensorConversion};
  std::vector<std::optional<Tensor>> result;
  result.reserve(objs.size());
  for (const auto& obj : objs) {
    if (obj.is_none()) {
      result.push_back(std::nullopt);
    } else {
      result.push_back(detail::TensorFromPybind11HandleImpl(obj));
    }
  }
  return result;
}

}  // namespace infini::ops

#endif
