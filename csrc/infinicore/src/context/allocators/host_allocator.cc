#include "host_allocator.hpp"

namespace infinicore {
std::byte *HostAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    return (std::byte *)std::malloc(size);
}

void HostAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }
    std::free(ptr);
}

} // namespace infinicore
