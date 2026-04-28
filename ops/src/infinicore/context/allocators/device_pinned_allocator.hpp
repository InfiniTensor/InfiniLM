#pragma once

#include "memory_allocator.hpp"

#include "../context_impl.hpp"

#include <queue>

namespace infinicore {
class DevicePinnedHostAllocator : public MemoryAllocator {
public:
    explicit DevicePinnedHostAllocator(Device device);
    ~DevicePinnedHostAllocator();

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;

    void gc();

private:
    Device owner_;

    /// TODO: this is not thread-safe
    std::queue<std::byte *> gc_queue_;
};

} // namespace infinicore
