#include "runtime/memory.hpp"

#include <utility>

namespace infinilm::runtime {

Memory::Memory(std::byte *data,
               size_t size,
               Device device,
               Memory::Deleter deleter,
               bool pin_memory,
               bool graph_pinnable,
               std::weak_ptr<Runtime> allocation_owner)
    : data_{data}, size_{size}, device_{device}, deleter_{std::move(deleter)},
      is_pinned_(pin_memory), graph_pinnable_(graph_pinnable),
      allocation_owner_(std::move(allocation_owner)) {}

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

bool Memory::is_graph_pinnable() const {
    return graph_pinnable_;
}
} // namespace infinilm::runtime
