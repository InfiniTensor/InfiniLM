#pragma once

#include "infinicore/memory.hpp"

#include <memory>

namespace infinicore {
class MemoryAllocator {
public:
    virtual ~MemoryAllocator() = default;

    virtual std::byte *allocate(size_t size) = 0;
    virtual void deallocate(std::byte *ptr) = 0;
};
} // namespace infinicore
