#pragma once

#include "memory_allocator.hpp"

#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace infinicore {
class PinnableBlockAllocator : public MemoryAllocator {
    // Represents a single memory block
    struct Block {
        void *ptr = nullptr;  // Device pointer
        size_t size = 0;      // Block size in bytes
        size_t pin_count = 0; // Number of graphs retaining this block
        bool in_use = false;  // Wether the block is currently in use
        size_t use_count = 0; // Number of Memory owners for this block
    };

    // A simple size-class allocator for small/medium blocks
    struct SizeClass {
        size_t block_size; // Fixed size for this class
        std::vector<std::shared_ptr<Block>> free_blocks;
    };

public:
    class PinLease {
    public:
        PinLease(PinnableBlockAllocator *allocator,
                 std::vector<std::shared_ptr<Block>> blocks) noexcept;
        ~PinLease() noexcept;

        PinLease(const PinLease &) = delete;
        PinLease &operator=(const PinLease &) = delete;

    private:
        PinnableBlockAllocator *allocator_;
        std::vector<std::shared_ptr<Block>> blocks_;
    };

    PinnableBlockAllocator(Device device);
    ~PinnableBlockAllocator();

    std::byte *allocate(size_t size) override;
    void deallocate(std::byte *ptr) override;

    // Manage one graph-capture pin transaction.
    void begin_pin_mode();
    std::shared_ptr<PinLease> commit_pin_mode();
    void cancel_pin_mode();
    void retain_for_capture(void *ptr);

    // internal use only, force set in_use flag for a mem block
    // return the size of the block
    size_t mark_in_use_(void *ptr, bool in_use);

    // trim cached blocks back to GPU (not pinned)
    void trim();

private:
    Device device_;

    bool pinned_mode_ = false;
    std::thread::id pin_owner_;

    std::vector<SizeClass> size_classes_;
    std::vector<std::shared_ptr<Block>> large_blocks_;
    std::unordered_map<void *, std::shared_ptr<Block>> all_blocks_;
    std::vector<std::shared_ptr<Block>> capture_frozen_blocks_;
    std::unordered_set<Block *> capture_frozen_block_set_;

    std::mutex mutex_; // Thread safety

    void freeze_for_capture_(const std::shared_ptr<Block> &block);
    void release_frozen_blocks_(
        const std::vector<std::shared_ptr<Block>> &blocks) noexcept;
};

} // namespace infinicore
