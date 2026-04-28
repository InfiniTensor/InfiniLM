#include "pinnable_block_allocator.hpp"

#include "../context_impl.hpp"

#include "../../utils.hpp"

#include <algorithm>
#include <infinirt.h>
#include <stdexcept>

namespace infinicore {

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
            if (!cls.free_blocks.empty()) {
                // Skip blocks that are either still in use or frozen-but-not-
                // pinned-mode-now: a frozen block was given to a captured
                // CUDA graph and must NOT be reused for non-graph allocations
                // until either pinned_mode_ resumes (same graph capture cycle
                // can reuse them safely) or trim() releases them.
                size_t scan = cls.free_blocks.size();
                block = nullptr;
                while (scan > 0) {
                    --scan;
                    auto &candidate = cls.free_blocks[scan];
                    if (!candidate->in_use && (!candidate->frozen || pinned_mode_)) {
                        block = candidate;
                        cls.free_blocks.erase(cls.free_blocks.begin() + scan);
                        break;
                    }
                }
                if (block != nullptr) {
                    block->in_use = true;
                    // Freeze is sticky: once a block is handed to a captured graph
                    // it stays frozen until trim() releases it.
                    if (pinned_mode_) {
                        block->frozen = true;
                    }
                    return reinterpret_cast<std::byte *>(block->ptr);
                }
            }
            // Allocate a new block for this class
            block = std::make_shared<Block>();
            block->size = cls.block_size;
            block->frozen = pinned_mode_;
            block->in_use = true;

            INFINICORE_CHECK_ERROR(infinirtMalloc(&block->ptr, block->size));

            all_blocks_[block->ptr] = block;
            return reinterpret_cast<std::byte *>(block->ptr);
        }
    }

    // 2. Large block allocation
    // Reuse a free non-frozen large block (frozen blocks belong to a captured
    // CUDA graph and must be skipped unless we're currently in pinned mode).
    auto it = std::find_if(large_blocks_.begin(), large_blocks_.end(),
                           [this, size](const std::shared_ptr<Block> &b) {
                               return b->size >= size && !b->in_use && (!b->frozen || pinned_mode_);
                           });

    if (it != large_blocks_.end()) {
        block = *it;
        block->in_use = true;
        if (pinned_mode_) {
            block->frozen = true;
        }
        return reinterpret_cast<std::byte *>(block->ptr);
    }

    // Allocate new large block
    block = std::make_shared<Block>();
    block->size = size;
    block->frozen = pinned_mode_;
    block->in_use = true;

    INFINICORE_CHECK_ERROR(infinirtMalloc(&block->ptr, block->size));

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
    if (!block->in_use) {
        throw std::runtime_error("Double free detected in PinnableBlockAllocator");
    }

    block->in_use = false;

    if (!block->in_use) {
        for (auto &cls : size_classes_) {
            if (block->size == cls.block_size) {
                cls.free_blocks.push_back(block);
                break;
            }
        }
    }
}

size_t PinnableBlockAllocator::mark_in_use_(void *ptr, bool in_use) {
    auto it = all_blocks_.find(reinterpret_cast<void *>(ptr));
    if (it == all_blocks_.end()) {
        throw std::runtime_error("Pointer not allocated by this allocator");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    it->second->in_use = in_use;
    return it->second->size;
}

// ------------------- trim -------------------
void PinnableBlockAllocator::trim() {
    std::lock_guard<std::mutex> lock(mutex_);
    // Free non-frozen size-class blocks
    for (auto &cls : size_classes_) {
        for (auto it = cls.free_blocks.begin(); it != cls.free_blocks.end();) {
            if (!(*it)->frozen) {
                INFINICORE_CHECK_ERROR(infinirtFree((*it)->ptr));
                all_blocks_.erase((*it)->ptr);
                it = cls.free_blocks.erase(it);
            } else {
                ++it;
            }
        }
    }
    // Free non-frozen large blocks
    for (auto it = large_blocks_.begin(); it != large_blocks_.end();) {
        if (!(*it)->frozen && !(*it)->in_use) {
            INFINICORE_CHECK_ERROR(infinirtFree((*it)->ptr));
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
            infinirtFree(p.second->ptr);
        }
    }
    all_blocks_.clear();
    large_blocks_.clear();
    for (auto &cls : size_classes_) {
        cls.free_blocks.clear();
    }
}

} // namespace infinicore
