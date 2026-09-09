#include "stream_ordered_allocator.hpp"

#include "../../utils.hpp"

namespace infinicore {
StreamOrderedAllocator::StreamOrderedAllocator(Device device) : MemoryAllocator(), device_(device) {}

std::byte *StreamOrderedAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    if (device_.type() != Device::Type::kCpu) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::MallocAsync(
            &ptr,
            size,
            context::getStream()));
    } else {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::Malloc(&ptr, size));
    }
    return (std::byte *)ptr;
}

void StreamOrderedAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }
    if (device_.type() != Device::Type::kCpu) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::FreeAsync(
            ptr,
            context::getStream()));
    } else {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::Free(ptr));
    }
}
} // namespace infinicore
