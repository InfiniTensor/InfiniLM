#include "device_pinned_allocator.hpp"

#include "../../utils.hpp"

namespace infinicore {
DevicePinnedHostAllocator::DevicePinnedHostAllocator(Device device) : MemoryAllocator(), owner_(device) {}

DevicePinnedHostAllocator::~DevicePinnedHostAllocator() noexcept {
    try {
        gc();
    } catch (const std::exception &error) {
        try {
            spdlog::warn("freeing pinned host memory failed during allocator cleanup: {}", error.what());
        } catch (...) {
        }
    } catch (...) {
        try {
            spdlog::warn("freeing pinned host memory failed during allocator cleanup");
        } catch (...) {
        }
    }
}

std::byte *DevicePinnedHostAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr;
    INFINICORE_CHECK_ERROR(infini::rt::runtime::MallocHost(&ptr, size));
    return (std::byte *)ptr;
}

void DevicePinnedHostAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }
    if (owner_ == context::getDevice()) {
        INFINICORE_CHECK_ERROR(infini::rt::runtime::FreeHost(ptr));
        gc();
    } else {
        std::lock_guard<std::mutex> lock{gc_mutex_};
        gc_queue_.push(ptr);
    }
}

void DevicePinnedHostAllocator::gc() {
    std::lock_guard<std::mutex> lock{gc_mutex_};
    while (gc_queue_.empty() == false) {
        std::byte *p = gc_queue_.front();
        INFINICORE_CHECK_ERROR(infini::rt::runtime::FreeHost(p));
        gc_queue_.pop();
    }
}

} // namespace infinicore
