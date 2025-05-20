#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include "infinicore_infer.h"

class AllocatorBase {
public:
    virtual void *alloc(size_t size) = 0;
    virtual void release(void *ptr) = 0;
};

class WorkspaceAllocator : public AllocatorBase {
private:
    void *_memory;
    size_t _total_size;
    size_t _used_size;
    size_t _align = 256;

public:
    WorkspaceAllocator(size_t intial_size, size_t align = 256);
    ~WorkspaceAllocator();
    void *alloc(size_t size) override;
    void release(void *ptr) override;
};

#endif
