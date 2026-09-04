#include "infinicore/memory.hpp"

namespace infinicore {

Memory::Memory(std::byte *data,
               size_t size,
               Device device,
               Memory::Deleter deleter,
               bool pin_memory)
    : data_{data}, size_{size}, device_{device}, deleter_{deleter}, is_pinned_(pin_memory) {}

Memory::~Memory() {
    if (deleter_) {
        deleter_(data_);
    }
}

std::byte *Memory::data() {
    return data_;
}

Device Memory::device() const {
    return device_;
}

size_t Memory::size() const {
    return size_;
}

bool Memory::is_pinned() const {
    return is_pinned_;
}
} // namespace infinicore
