#include "../allocator.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

MemoryPool::MemoryPool(size_t initialSize) {
    allocateNewRegion(initialSize);
}

MemoryPool::~MemoryPool() {
    for (void *region : _base_regions) {
        RUN_INFINI(infinirtFree(region));
    }
}

void *MemoryPool::alloc(size_t size) {
    if (size == 0) {
        return nullptr;
    }

    auto it = _free_blocks.lower_bound(size);
    if (it == _free_blocks.end()) {
        allocateNewRegion(size);
        it = _free_blocks.lower_bound(size);
        if (it == _free_blocks.end()) {
            throw std::bad_alloc();
        }
    }

    auto block_it = it->second;
    Block block = *block_it;
    _free_blocks.erase(it);
    _all_blocks.erase(block_it);

    if (block.size > size + 256) {
        // Split
        void *alloc_ptr = block.ptr;
        void *rem_ptr = static_cast<char *>(block.ptr) + size;
        size_t rem_size = block.size - size;
        Block alloc_block(block.base, alloc_ptr, size, false);
        Block rem_block(block.base, rem_ptr, rem_size, true);
        auto alloc_it = _all_blocks.insert(alloc_block).first;
        auto rem_it = _all_blocks.insert(rem_block).first;
        _free_blocks.emplace(rem_size, rem_it);
        _ptr_to_block[alloc_ptr] = alloc_it;
        return alloc_ptr;
    } else {
        // No split
        block.is_free = false;
        auto alloc_it = _all_blocks.insert(block).first;
        _ptr_to_block[block.ptr] = alloc_it;
        return block.ptr;
    }
}

void MemoryPool::release(void *ptr) {
    if (ptr == nullptr) {
        return;
    }

    auto it = _ptr_to_block.find(ptr);
    if (it == _ptr_to_block.end()) {
        throw std::runtime_error("Invalid pointer to free");
    }

    auto block_it = it->second;
    Block block = *block_it;
    _all_blocks.erase(block_it);
    block.is_free = true;
    auto new_it = _all_blocks.insert(block).first;
    _ptr_to_block.erase(ptr);
    tryCoalesce(*new_it);
}

void *MemoryPool::allocateNewRegion(size_t size) {
    void *ptr = nullptr;
    RUN_INFINI(infinirtMalloc(&ptr, size));
    _base_regions.push_back(ptr);
    Block new_block(ptr, ptr, size, true);
    auto it = _all_blocks.insert(new_block).first;
    _free_blocks.emplace(size, it);
    return ptr;
}

void MemoryPool::tryCoalesce(const Block &block) {
    auto it = _all_blocks.find(block);
    if (it == _all_blocks.end()) {
        return;
    }

    Block merged = *it;
    auto next = std::next(it);
    auto prev = (it == _all_blocks.begin()) ? _all_blocks.end() : std::prev(it);

    _all_blocks.erase(it);
    _free_blocks.erase(merged.size);

    // Coalesce with next
    if (next != _all_blocks.end() && next->is_free && static_cast<char *>(merged.ptr) + merged.size == next->ptr) {
        _free_blocks.erase(next->size);
        merged.size += next->size;
        _all_blocks.erase(next);
    }

    // Coalesce with prev
    if (prev != _all_blocks.end() && prev->is_free && static_cast<char *>(prev->ptr) + prev->size == merged.ptr) {
        _free_blocks.erase(prev->size);
        merged.ptr = prev->ptr;
        merged.size += prev->size;
        merged.base = prev->base;
        _all_blocks.erase(prev);
    }

    merged.is_free = true;
    auto new_it = _all_blocks.insert(merged).first;
    _free_blocks.emplace(merged.size, new_it);
}
