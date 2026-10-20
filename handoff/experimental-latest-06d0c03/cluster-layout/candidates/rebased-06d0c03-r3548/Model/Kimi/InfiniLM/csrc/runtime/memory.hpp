#pragma once

#include "device.hpp"

#include <cstddef>
#include <functional>
#include <memory>

namespace infinilm::runtime {
class Runtime;

class Memory {
public:
    using Deleter = std::function<void(std::byte *)>;

    Memory(std::byte *data,
           size_t size,
           Device device,
           Deleter deleter,
           bool pin_memory = false,
           bool graph_pinnable = false,
           std::weak_ptr<Runtime> allocation_owner = {});
    ~Memory();

    std::byte *data();
    Device device() const;
    size_t size() const;
    bool is_pinned() const;
    bool is_graph_pinnable() const;
    // Allocation identity only, not a last-writer or producer-stream tracker.
    std::shared_ptr<Runtime> allocation_owner() const { return allocation_owner_.lock(); }

private:
    std::byte *data_;
    size_t size_;
    Device device_;
    Deleter deleter_;
    bool is_pinned_;
    bool graph_pinnable_;
    std::weak_ptr<Runtime> allocation_owner_;
};

} // namespace infinilm::runtime
