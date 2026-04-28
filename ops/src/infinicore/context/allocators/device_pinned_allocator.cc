#include "device_pinned_allocator.hpp"

#include <infinirt.h>

#include "../../utils.hpp"

namespace infinicore {
DevicePinnedHostAllocator::DevicePinnedHostAllocator(Device device) : MemoryAllocator(), owner_(device) {}

DevicePinnedHostAllocator::~DevicePinnedHostAllocator() {
    gc();
}

std::byte *DevicePinnedHostAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr;
    INFINICORE_CHECK_ERROR(infinirtMallocHost(&ptr, size));
    return (std::byte *)ptr;
}

void DevicePinnedHostAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }
    if (owner_ == context::getDevice()) {
        INFINICORE_CHECK_ERROR(infinirtFreeHost(ptr));
        gc();
    } else {
        gc_queue_.push(ptr);
    }
}

void DevicePinnedHostAllocator::gc() {
    while (gc_queue_.empty() == false) {
        std::byte *p = gc_queue_.front();
        INFINICORE_CHECK_ERROR(infinirtFreeHost(p));
        gc_queue_.pop();
    }
}

} // namespace infinicore
