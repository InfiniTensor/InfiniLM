#include "../allocator.hpp"

#include "../utils.hpp"

inline size_t aligned_size(size_t size_, size_t align) {
    return (size_ + align - 1) & ~(align - 1);
}

inline void *allocate(size_t size_) {
    void *ptr;
    RUN_INFINI(infinirtMalloc(&ptr, size_));
    return ptr;
}

WorkspaceAllocator::WorkspaceAllocator(size_t initial_size_, size_t align) {
    _align = align;
    if (initial_size_ > 0) {
        _total_size = aligned_size(initial_size_, _align);
        _memory = allocate(_total_size);
    }
}

void *WorkspaceAllocator::alloc(size_t new_size) {
    if (_total_size < new_size) {
        if (_total_size != 0) {
            RUN_INFINI(infinirtFree(_memory));
        }
        _total_size = aligned_size(new_size * 3 / 2, _align);
        _memory = allocate(_total_size);
    }
    return _memory;
}

void WorkspaceAllocator::release(void *ptr) {
}

WorkspaceAllocator::~WorkspaceAllocator() {
    if (_memory != nullptr) {
        RUN_INFINI(infinirtFree(_memory));
    }
}