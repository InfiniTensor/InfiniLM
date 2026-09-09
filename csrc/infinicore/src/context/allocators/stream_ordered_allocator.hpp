#pragma once

#include "memory_allocator.hpp"

#include "../context_impl.hpp"

namespace infinicore {
class StreamOrderedAllocator : public MemoryAllocator {
public:
    explicit StreamOrderedAllocator(Device device);
    ~StreamOrderedAllocator() = default;

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;

private:
    Device device_;
};

} // namespace infinicore
