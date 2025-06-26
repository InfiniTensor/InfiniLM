#include "../allocator.hpp"
#include "../utils.hpp"

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
