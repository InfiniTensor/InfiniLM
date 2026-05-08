#include "stream_ordered_allocator.hpp"

#include <infinirt.h>

#include "../../utils.hpp"

namespace infinicore {
StreamOrderedAllocator::StreamOrderedAllocator(Device device) : MemoryAllocator(), device_(device) {}

std::byte *StreamOrderedAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    INFINICORE_CHECK_ERROR(infinirtMallocAsync(&ptr, size, context::getStream()));
    return (std::byte *)ptr;
}

void StreamOrderedAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }
    INFINICORE_CHECK_ERROR(infinirtFreeAsync(ptr, context::getStream()));
}
} // namespace infinicore
