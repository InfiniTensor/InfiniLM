#include "../allocator.hpp"
#include "../utils.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>

MemoryPool::MemoryPool(size_t initialSize) {
    allocateNewRegion(initialSize);
}

MemoryPool::~MemoryPool() {
    for (void *region : baseRegions) {
        RUN_INFINI(infinirtFree(region));
    }
}

void *MemoryPool::alloc(size_t size) {
    auto it = freeBlocks.lower_bound(size);
    if (it == freeBlocks.end()) {
        allocateNewRegion(std::max(size, size_t(0)));
        it = freeBlocks.lower_bound(size);
        if (it == freeBlocks.end()) {
            throw std::bad_alloc();
        }
    }

    auto blockIt = it->second;
    Block block = *blockIt;
    freeBlocks.erase(it);
    allBlocks.erase(blockIt);

    if (block.size > size + 256) {
        // Split
        void *allocPtr = block.ptr;
        void *remPtr = static_cast<char *>(block.ptr) + size;
        size_t remSize = block.size - size;
        Block allocBlock(block.base, allocPtr, size, false);
        Block remBlock(block.base, remPtr, remSize, true);
        auto allocIt = allBlocks.insert(allocBlock).first;
        auto remIt = allBlocks.insert(remBlock).first;
        freeBlocks.emplace(remSize, remIt);
        ptrToBlock[allocPtr] = allocIt;
        return allocPtr;
    } else {
        // No split
        block.isFree = false;
        auto allocIt = allBlocks.insert(block).first;
        ptrToBlock[block.ptr] = allocIt;
        return block.ptr;
    }
}

void MemoryPool::release(void *ptr) {
    auto it = ptrToBlock.find(ptr);
    if (it == ptrToBlock.end()) {
        throw std::runtime_error("Invalid pointer to free");
    }

    auto blockIt = it->second;
    Block block = *blockIt;
    allBlocks.erase(blockIt);
    block.isFree = true;
    auto newIt = allBlocks.insert(block).first;
    ptrToBlock.erase(ptr);
    tryCoalesce(*newIt);
}

void *MemoryPool::allocateNewRegion(size_t size) {
    void *ptr = nullptr;
    RUN_INFINI(infinirtMalloc(&ptr, size));
    baseRegions.push_back(ptr);
    Block newBlock(ptr, ptr, size, true);
    auto it = allBlocks.insert(newBlock).first;
    freeBlocks.emplace(size, it);
    return ptr;
}

void MemoryPool::tryCoalesce(const Block &block) {
    auto it = allBlocks.find(block);
    if (it == allBlocks.end()) {
        return;
    }

    Block merged = *it;
    auto next = std::next(it);
    auto prev = (it == allBlocks.begin()) ? allBlocks.end() : std::prev(it);

    allBlocks.erase(it);
    freeBlocks.erase(merged.size);

    // Coalesce with next
    if (next != allBlocks.end() && next->isFree && static_cast<char *>(merged.ptr) + merged.size == next->ptr) {
        freeBlocks.erase(next->size);
        merged.size += next->size;
        allBlocks.erase(next);
    }

    // Coalesce with prev
    if (prev != allBlocks.end() && prev->isFree && static_cast<char *>(prev->ptr) + prev->size == merged.ptr) {
        freeBlocks.erase(prev->size);
        merged.ptr = prev->ptr;
        merged.size += prev->size;
        merged.base = prev->base;
        allBlocks.erase(prev);
    }

    merged.isFree = true;
    auto newIt = allBlocks.insert(merged).first;
    freeBlocks.emplace(merged.size, newIt);
}

WorkspaceHandle::WorkspaceHandle(std::shared_ptr<MemoryPool> pool, size_t size)
    : pool(pool), size(size) {
    ptr = pool->alloc(size);
    if (!ptr) {
        throw std::runtime_error("Failed to allocate workspace");
    }
}

WorkspaceHandle::~WorkspaceHandle() {
    if (ptr) {
        pool->release(ptr);
    }
}
