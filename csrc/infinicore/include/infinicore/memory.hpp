#pragma once

#include "device.hpp"

#include <cstddef>
#include <functional>

namespace infinicore {

class Memory {
public:
    using Deleter = std::function<void(std::byte *)>;

    Memory(std::byte *data, size_t size, Device device, Deleter deleter, bool pin_memory = false);
    ~Memory();

    std::byte *data();
    Device device() const;
    size_t size() const;
    bool is_pinned() const;

private:
    std::byte *data_;
    size_t size_;
    Device device_;
    Deleter deleter_;
    bool is_pinned_;
};

} // namespace infinicore
