#pragma once

#include "memory_allocator.hpp"

namespace infinicore {
class HostAllocator : public MemoryAllocator {
public:
    HostAllocator() = default;
    ~HostAllocator() = default;

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;
};

} // namespace infinicore
