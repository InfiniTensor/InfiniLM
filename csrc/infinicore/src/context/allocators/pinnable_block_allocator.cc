#include "pinnable_block_allocator.hpp"

#include "../context_impl.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace infinicore {

PinnableBlockAllocator::PinLease::PinLease(
    PinnableBlockAllocator *allocator,
    std::vector<std::shared_ptr<Block>> blocks) noexcept
    : allocator_(allocator), blocks_(std::move(blocks)) {}

PinnableBlockAllocator::PinLease::~PinLease() noexcept {
    allocator_->release_frozen_blocks_(blocks_);
}

// ------------------- Helper functions -------------------

// Round up size to nearest multiple of alignment
inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

// ------------------- Constructor -------------------
PinnableBlockAllocator::PinnableBlockAllocator(Device device)
    : device_(device) {
    size_classes_ = {
        {32 * 1024, {}},         // 32 KB
        {256 * 1024, {}},        // 256 KB
        {1 * 1024 * 1024, {}},   // 1 MB
        {2 * 1024 * 1024, {}},   // 2 MB
        {4 * 1024 * 1024, {}},   // 4 MB
        {8 * 1024 * 1024, {}},   // 8 MB
        {16 * 1024 * 1024, {}},  // 16 MB
        {32 * 1024 * 1024, {}},  // 32 MB
        {64 * 1024 * 1024, {}},  // 64 MB
        {128 * 1024 * 1024, {}}, // 128 MB
        {256 * 1024 * 1024, {}}, // 256 MB
    };
}

void PinnableBlockAllocator::begin_pin_mode() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (pinned_mode_) {
        throw std::runtime_error("allocator graph capture is already active");
    }
    INFINICORE_ASSERT(capture_frozen_blocks_.empty());
    INFINICORE_ASSERT(capture_frozen_block_set_.empty());
    pinned_mode_ = true;
    pin_owner_ = std::this_thread::get_id();
}

std::shared_ptr<PinnableBlockAllocator::PinLease>
PinnableBlockAllocator::commit_pin_mode() {
    std::vector<std::shared_ptr<Block>> blocks;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pinned_mode_) {
            return nullptr;
        }
        if (pin_owner_ != std::this_thread::get_id()) {
            throw std::runtime_error("cannot commit allocator pin mode: another thread owns the graph capture");
        }
        pinned_mode_ = false;
        pin_owner_ = {};
        blocks = std::move(capture_frozen_blocks_);
        capture_frozen_block_set_.clear();
    }

    try {
        return std::make_shared<PinLease>(this, std::move(blocks));
    } catch (...) {
        release_frozen_blocks_(blocks);
        throw;
    }
}

void PinnableBlockAllocator::cancel_pin_mode() {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto caller = std::this_thread::get_id();
    if (pinned_mode_ && pin_owner_ != caller) {
        throw std::runtime_error("cannot cancel allocator pin mode: another thread owns the graph capture");
    }
    for (const auto &block : capture_frozen_blocks_) {
        if (block->pin_count > 0) {
            --block->pin_count;
        }
    }
    capture_frozen_blocks_.clear();
    capture_frozen_block_set_.clear();
    pinned_mode_ = false;
    pin_owner_ = {};
}

void PinnableBlockAllocator::freeze_for_capture_(const std::shared_ptr<Block> &block) {
    if (pinned_mode_ && pin_owner_ == std::this_thread::get_id()
        && capture_frozen_block_set_.insert(block.get()).second) {
        ++block->pin_count;
        capture_frozen_blocks_.push_back(block);
    }
}

void PinnableBlockAllocator::retain_for_capture(void *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pinned_mode_ || pin_owner_ != std::this_thread::get_id()) {
        throw std::runtime_error("allocator graph capture is not active on this thread");
    }

    const auto it = all_blocks_.find(ptr);
    if (it == all_blocks_.end()) {
        throw std::runtime_error(
            "graph capture requires tensors backed by InfiniCore-owned memory");
    }
    freeze_for_capture_(it->second);
}

void PinnableBlockAllocator::release_frozen_blocks_(
    const std::vector<std::shared_ptr<Block>> &blocks) noexcept {
    try {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto &block : blocks) {
            if (block->pin_count > 0) {
                --block->pin_count;
            }
        }
    } catch (...) {
    }
}

// ------------------- allocate -------------------
std::byte *PinnableBlockAllocator::allocate(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // Align size to 256 bytes for GPU
    size = align_up(size, 256);

    std::shared_ptr<Block> block;

    // 1. Try size-class allocation for small/medium
    for (auto &cls : size_classes_) {
        if (size <= cls.block_size) {
            const auto free_block = std::find_if(
                cls.free_blocks.begin(), cls.free_blocks.end(),
                [](const auto &block) {
                    return !block->in_use && block->pin_count == 0;
                });
            if (free_block != cls.free_blocks.end()) {
                block = *free_block;
                cls.free_blocks.erase(free_block);
                block->in_use = true;
                block->use_count = 1;
                freeze_for_capture_(block);
                return reinterpret_cast<std::byte *>(block->ptr);
            }
            // Allocate a new block for this class
            block = std::make_shared<Block>();
            block->size = cls.block_size;
            block->in_use = true;
            block->use_count = 1;
            freeze_for_capture_(block);

            INFINICORE_CHECK_ERROR(infini::rt::runtime::Malloc(&block->ptr, block->size));

            all_blocks_[block->ptr] = block;
            return reinterpret_cast<std::byte *>(block->ptr);
        }
    }

    // 2. Large block allocation
    // Try to reuse an unpinned free large block.
    auto it = std::find_if(large_blocks_.begin(), large_blocks_.end(),
                           [size](const std::shared_ptr<Block> &b) {
                               return b->size >= size && !b->in_use && b->pin_count == 0;
                           });

    if (it != large_blocks_.end()) {
        block = *it;
        block->in_use = true;
        block->use_count = 1;
        freeze_for_capture_(block);
        return reinterpret_cast<std::byte *>(block->ptr);
    }

    // Allocate new large block
    block = std::make_shared<Block>();
    block->size = size;
    block->in_use = true;
    block->use_count = 1;
    freeze_for_capture_(block);

    INFINICORE_CHECK_ERROR(infini::rt::runtime::Malloc(&block->ptr, block->size));

    large_blocks_.push_back(block);
    all_blocks_[block->ptr] = block;

    return reinterpret_cast<std::byte *>(block->ptr);
}

// ------------------- deallocate -------------------
void PinnableBlockAllocator::deallocate(std::byte *ptr) {
    if (ptr == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = all_blocks_.find(reinterpret_cast<void *>(ptr));
    if (it == all_blocks_.end()) {
        throw std::runtime_error("Pointer not allocated by this allocator");
    }

    auto block = it->second;
    if (!block->in_use || block->use_count == 0) {
        throw std::runtime_error("Double free detected in PinnableBlockAllocator");
    }

    --block->use_count;
    if (block->use_count > 0) {
        return;
    }

    block->in_use = false;
    for (auto &cls : size_classes_) {
        if (block->size == cls.block_size) {
            cls.free_blocks.push_back(block);
            break;
        }
    }
}

size_t PinnableBlockAllocator::mark_in_use_(void *ptr, bool in_use) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = all_blocks_.find(reinterpret_cast<void *>(ptr));
    if (it == all_blocks_.end()) {
        throw std::runtime_error("Pointer not allocated by this allocator");
    }

    auto block = it->second;
    if (in_use) {
        for (auto &cls : size_classes_) {
            if (block->size == cls.block_size) {
                cls.free_blocks.erase(
                    std::remove(
                        cls.free_blocks.begin(),
                        cls.free_blocks.end(),
                        block),
                    cls.free_blocks.end());
                break;
            }
        }
        block->in_use = true;
        ++block->use_count;
    } else if (block->use_count > 0) {
        --block->use_count;
        block->in_use = block->use_count > 0;
    }
    return it->second->size;
}

// ------------------- trim -------------------
void PinnableBlockAllocator::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Free non-frozen size-class blocks
    for (auto &cls : size_classes_) {
        for (auto it = cls.free_blocks.begin(); it != cls.free_blocks.end();) {
            if ((*it)->pin_count == 0) {
                INFINICORE_CHECK_ERROR(infini::rt::runtime::Free((*it)->ptr));
                all_blocks_.erase((*it)->ptr);
                it = cls.free_blocks.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Free non-frozen large blocks
    for (auto it = large_blocks_.begin(); it != large_blocks_.end();) {
        if ((*it)->pin_count == 0 && !(*it)->in_use) {
            INFINICORE_CHECK_ERROR(infini::rt::runtime::Free((*it)->ptr));
            all_blocks_.erase((*it)->ptr);
            it = large_blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

// ------------------- Destructor -------------------
PinnableBlockAllocator::~PinnableBlockAllocator() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &p : all_blocks_) {
        if (p.second->ptr) {
            (void)infini::rt::runtime::Free(p.second->ptr);
        }
    }
    all_blocks_.clear();
    large_blocks_.clear();
    for (auto &cls : size_classes_) {
        cls.free_blocks.clear();
    }
}

} // namespace infinicore
