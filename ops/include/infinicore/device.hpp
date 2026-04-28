#pragma once

#include <cstdint>
#include <string>

#include "infinicore.h"

namespace infinicore {

class Device {
public:
    using Index = std::size_t;

    enum class Type {
        CPU = INFINI_DEVICE_CPU,
        NVIDIA = INFINI_DEVICE_NVIDIA,
        CAMBRICON = INFINI_DEVICE_CAMBRICON,
        ASCEND = INFINI_DEVICE_ASCEND,
        METAX = INFINI_DEVICE_METAX,
        MOORE = INFINI_DEVICE_MOORE,
        ILUVATAR = INFINI_DEVICE_ILUVATAR,
        KUNLUN = INFINI_DEVICE_KUNLUN,
        HYGON = INFINI_DEVICE_HYGON,
        QY = INFINI_DEVICE_QY,
        ALI = INFINI_DEVICE_ALI,
        COUNT = INFINI_DEVICE_TYPE_COUNT,
    };

    Device(const Type &type = Type::CPU, const Index &index = 0);

    const Type &getType() const;

    const Index &getIndex() const;

    std::string toString() const;

    static std::string toString(const Type &type);

    bool operator==(const Device &other) const;

    bool operator!=(const Device &other) const;

    inline static Device cpu() {
        return Device(Type::CPU, 0);
    }

private:
    Type type_;

    Index index_;
};

} // namespace infinicore
