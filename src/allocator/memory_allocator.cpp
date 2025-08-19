#include "../allocator.hpp"
#include "../utils.hpp"
#include <vector>

MemoryPool::MemoryPool(size_t initialSize, size_t alignment)
    : _alignment(alignment) {
    // Validate alignment is power of two
    if ((alignment & (alignment - 1)) != 0 || alignment == 0) {
        throw std::invalid_argument("Alignment must be a power of two");
    }

    if (initialSize > 0) {
        allocateNewRegion(initialSize);
    }
}

MemoryPool::~MemoryPool() {
    for (void *region : _base_regions) {
        RUN_INFINI(infinirtFree(region));
    }
}
void MemoryPool::reset() {
    // This function reuses the existing `release()` logic for all allocated blocks.

    // 1. Create a temporary vector of all currently allocated pointers.
    //    We do this because the `release` function will modify the `_ptr_to_block` map
    //    we are iterating over, which would invalidate our iterator.
    std::vector<void *> ptrs_to_release;
    ptrs_to_release.reserve(_ptr_to_block.size());
    for (const auto& pair : _ptr_to_block) {
        ptrs_to_release.push_back(pair.first);
    }

    // 2. Call the existing `release()` function for each allocated block.
    //    This will automatically trigger the `tryCoalesce` logic and merge
    //    all memory back into the largest possible free blocks.
    for (void* ptr : ptrs_to_release) {
        release(ptr);
    }
}

void *MemoryPool::alloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    // Calculate aligned size
    const size_t aligned_size = (size + _alignment - 1) & ~(_alignment - 1);

    // Find the first block with enough space (after alignment)
    auto it = _free_blocks.lower_bound(aligned_size);
    if (it == _free_blocks.end()) {
        allocateNewRegion(aligned_size);
        it = _free_blocks.lower_bound(aligned_size);
        if (it == _free_blocks.end()) {
            throw std::bad_alloc();
        }
    }

    auto block_it = it->second;
    Block block = *block_it;
    _free_blocks.erase(it);
    _all_blocks.erase(block_it);

    // Align the pointer within the block
    size_t alignment_padding = reinterpret_cast<char *>(block.ptr) - reinterpret_cast<char *>(block.ptr);

    // Calculate remaining space after allocation
    const size_t remaining = block.size - aligned_size - alignment_padding;

    // Create allocated block
    Block alloc_block(block.base, block.ptr, aligned_size, false);
    auto alloc_it = _all_blocks.insert(alloc_block).first;
    _ptr_to_block[block.ptr] = alloc_it;

    // Split remaining space if it's large enough
    if (remaining >= _alignment) {
        void *rem_ptr = static_cast<char *>(block.ptr) + aligned_size;
        Block rem_block(block.base, rem_ptr, remaining, true);
        auto rem_it = _all_blocks.insert(rem_block).first;
        _free_blocks.emplace(remaining, rem_it);
    }

    return block.ptr;
}

void *MemoryPool::allocateNewRegion(size_t size) {
    // Allocate exactly the requested size
    void *ptr = nullptr;
    RUN_INFINI(infinirtMalloc(&ptr, size));
    _base_regions.push_back(ptr);

    // Align the pointer within the allocated region
    size_t alignment_padding = reinterpret_cast<char *>(ptr) - reinterpret_cast<char *>(ptr);
    size_t usable_size = size - alignment_padding;

    Block new_block(ptr, ptr, usable_size, true);
    auto it = _all_blocks.insert(new_block).first;
    _free_blocks.emplace(usable_size, it);

    return ptr;
}

void MemoryPool::release(void *ptr) {
    if (ptr == nullptr) return;

    auto map_it = _ptr_to_block.find(ptr);
    if (map_it == _ptr_to_block.end()) {
        return; // Safe to ignore, might be from a reset call
    }

    auto block_it = map_it->second;
    if (block_it->is_free) {
        std::cerr << "WARNING: Double free detected." << std::endl;
        return;
    }

    // Mark the block as free and update its place in the master list
    Block block = *block_it;
    block.is_free = true;
    _all_blocks.erase(block_it);
    auto new_free_it = _all_blocks.insert(block).first;
    
    _ptr_to_block.erase(map_it);
    
    // Pass the iterator to the new free block to coalesce
    tryCoalesce(*new_free_it);
}

void MemoryPool::tryCoalesce(const Block &block) {
    auto block_it = _all_blocks.find(block);
    if (block_it == _all_blocks.end() || !block_it->is_free) {
        return;
    }

    Block merged_block = *block_it;
    
    // --- Check and merge with PREVIOUS block ---
    auto prev_it = _all_blocks.end(); // Use as a sentinel
    if (block_it != _all_blocks.begin()) {
        auto temp_prev = std::prev(block_it);
        if (temp_prev->is_free &&
            static_cast<char*>(temp_prev->ptr) + temp_prev->size == merged_block.ptr &&
            temp_prev->base == merged_block.base) {
            
            merged_block.ptr = temp_prev->ptr;
            merged_block.size += temp_prev->size;
            prev_it = temp_prev; // Mark previous block for erasure
        }
    }

    // --- Check and merge with NEXT block ---
    auto next_it = _all_blocks.end(); // Use as a sentinel
    auto temp_next = std::next(block_it);
    if (temp_next != _all_blocks.end() && temp_next->is_free &&
        static_cast<char*>(merged_block.ptr) + merged_block.size == temp_next->ptr &&
        merged_block.base == temp_next->base) {

        merged_block.size += temp_next->size;
        next_it = temp_next; // Mark next block for erasure
    }

    // --- Finalize the merge operation ---
    if (prev_it != _all_blocks.end() || next_it != _all_blocks.end()) {
        // Remove the original block from the free list (if it was added) and master list
        // Note: The original `release` doesn't add to free list, so we only need to erase from _all_blocks
        _all_blocks.erase(block_it);

        // Remove previous block if it was merged
        if (prev_it != _all_blocks.end()) {
            auto range = _free_blocks.equal_range(prev_it->size);
            for (auto it_free = range.first; it_free != range.second; ++it_free) {
                if (it_free->second == prev_it) { _free_blocks.erase(it_free); break; }
            }
            _all_blocks.erase(prev_it);
        }

        // Remove next block if it was merged
        if (next_it != _all_blocks.end()) {
            auto range = _free_blocks.equal_range(next_it->size);
            for (auto it_free = range.first; it_free != range.second; ++it_free) {
                if (it_free->second == next_it) { _free_blocks.erase(it_free); break; }
            }
            _all_blocks.erase(next_it);
        }

        // Insert the final, fully merged block and add it to the free list
        auto final_it = _all_blocks.insert(merged_block).first;
        _free_blocks.emplace(final_it->size, final_it);
    } else {
        // No merge happened, so just add the original single block to the free list.
        _free_blocks.emplace(block_it->size, block_it);
    }
}

