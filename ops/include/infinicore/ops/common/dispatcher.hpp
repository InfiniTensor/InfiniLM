#pragma once

#include "../../device.hpp"

#include <array>

namespace infinicore::op::common {
template <typename Fn>
class OpDispatcher {
public:
    void registerDevice(Device::Type device_type, Fn fn, bool override_existing = true) {
        if (table_[(size_t)device_type] == nullptr || override_existing) {
            table_[(size_t)device_type] = fn;
        }
    }

    void registerDevice(std::initializer_list<Device::Type> device_types, Fn fn, bool override_existing = true) {
        for (auto device_type : device_types) {
            registerDevice(device_type, fn, override_existing);
        }
    }

    void registerAll(Fn fn, bool override_existing = true) {
        for (size_t device_type = 0; device_type < static_cast<size_t>(Device::Type::COUNT); ++device_type) {
            registerDevice((Device::Type)device_type, fn, override_existing);
        }
    }

    Fn lookup(Device::Type device_type) const {
        return table_.at((size_t)device_type);
    }

private:
    std::array<Fn, static_cast<size_t>(Device::Type::COUNT)> table_;
};
} // namespace infinicore::op::common
