#pragma once

#include "../utils.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "config.h"
#include "data_type.h"
#include "handle.h"
#include "infini/ops.h"
#include "tensor.h"

namespace infinicore::op::infiniops {

inline infini::ops::DataType toInfiniOpsDtype(DataType dtype) {
    switch (dtype) {
    case DataType::kInt8:
        return infini::ops::DataType::kInt8;
    case DataType::kInt16:
        return infini::ops::DataType::kInt16;
    case DataType::kInt32:
        return infini::ops::DataType::kInt32;
    case DataType::kInt64:
        return infini::ops::DataType::kInt64;
    case DataType::kUInt8:
        return infini::ops::DataType::kUInt8;
    case DataType::kUInt16:
        return infini::ops::DataType::kUInt16;
    case DataType::kUInt32:
        return infini::ops::DataType::kUInt32;
    case DataType::kUInt64:
        return infini::ops::DataType::kUInt64;
    case DataType::kFloat16:
        return infini::ops::DataType::kFloat16;
    case DataType::kBFloat16:
        return infini::ops::DataType::kBFloat16;
    case DataType::kFloat32:
        return infini::ops::DataType::kFloat32;
    case DataType::kFloat64:
        return infini::ops::DataType::kFloat64;
    default:
        throw std::runtime_error("InfiniOps backend does not support this tensor dtype.");
    }
}

inline infini::ops::Device toInfiniOpsDevice(const Device &device) {
    switch (device.type()) {
    case Device::Type::kNvidia:
        return infini::ops::Device{infini::ops::Device::Type::kNvidia, static_cast<int>(device.index())};
    case Device::Type::kMetax:
        return infini::ops::Device{infini::ops::Device::Type::kMetax, static_cast<int>(device.index())};
    case Device::Type::kMoore:
        return infini::ops::Device{infini::ops::Device::Type::kMoore, static_cast<int>(device.index())};
    case Device::Type::kIluvatar:
        return infini::ops::Device{infini::ops::Device::Type::kIluvatar, static_cast<int>(device.index())};
    case Device::Type::kCambricon:
        return infini::ops::Device{infini::ops::Device::Type::kCambricon, static_cast<int>(device.index())};
    case Device::Type::kAscend:
        return infini::ops::Device{infini::ops::Device::Type::kAscend, static_cast<int>(device.index())};
    default:
        throw std::runtime_error("InfiniOps backend does not support this device type.");
    }
}

inline bool isSupportedDevice(Device::Type device_type) {
    switch (device_type) {
    case Device::Type::kNvidia:
    case Device::Type::kMetax:
    case Device::Type::kMoore:
    case Device::Type::kIluvatar:
    case Device::Type::kCambricon:
    case Device::Type::kAscend:
        return true;
    default:
        return false;
    }
}

template <typename Operator>
infini::ops::Config configForImplementation(
    infini::ops::Device::Type device_type,
    std::size_t implementation_index) {
    const auto implementation_indices = Operator::active_implementation_indices(device_type);
    if (std::find(
            implementation_indices.begin(),
            implementation_indices.end(),
            implementation_index)
        == implementation_indices.end()) {
        throw std::runtime_error(
            "InfiniOps implementation " + std::to_string(implementation_index)
            + " is not active for device '"
            + std::string(infini::ops::Device::StringFromType(device_type)) + "'.");
    }

    infini::ops::Config config;
    config.set_implementation_index(implementation_index);
    return config;
}

template <typename Operator>
infini::ops::Config defaultConfigForDevice(infini::ops::Device::Type device_type) {
    const auto implementation_indices = Operator::active_implementation_indices(device_type);
    if (implementation_indices.empty()) {
        throw std::runtime_error(
            "InfiniOps operator has no active implementation for device '"
            + std::string(infini::ops::Device::StringFromType(device_type)) + "'.");
    }

    return configForImplementation<Operator>(device_type, implementation_indices.front());
}

template <typename Dispatcher, typename Function>
void registerSupportedDevices(Dispatcher &dispatcher, Function function) {
    dispatcher.registerDevice(Device::Type::kNvidia, function);
    dispatcher.registerDevice(Device::Type::kMetax, function);
    dispatcher.registerDevice(Device::Type::kMoore, function);
    dispatcher.registerDevice(Device::Type::kIluvatar, function);
    dispatcher.registerDevice(Device::Type::kCambricon, function);
    dispatcher.registerDevice(Device::Type::kAscend, function);
}

struct TensorMeta {
    Shape shape;
    Strides strides;
    infini::ops::DataType dtype;
    infini::ops::Device device;

    explicit TensorMeta(const Tensor &tensor)
        : shape(tensor->shape()),
          strides(tensor->strides()),
          dtype(toInfiniOpsDtype(tensor->dtype())),
          device(toInfiniOpsDevice(tensor->device())) {}

    infini::ops::Tensor tensor(const void *data) const {
        return infini::ops::Tensor(const_cast<void *>(data), shape, dtype, device, strides);
    }

    infini::ops::Tensor tensor(const Tensor &tensor) const {
        return this->tensor(tensor->data());
    }
};

} // namespace infinicore::op::infiniops
