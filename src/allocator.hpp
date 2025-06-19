#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include "infinicore_infer.h"
#include <map>
#include <set>
#include <unordered_map>
#include <vector>

class AllocatorBase {
public:
    virtual void *alloc(size_t size) = 0;
    virtual void release(void *ptr) = 0;
    virtual ~AllocatorBase() = default;
};

class MemoryPool : public AllocatorBase {
public:
    MemoryPool(size_t initialSize = 32 * 1024 * 1024); // default: 32MB
    ~MemoryPool();
    void *alloc(size_t size) override;
    void release(void *ptr) override;

private:
    struct Block {
        void *base;
        void *ptr;
        size_t size;
        bool isFree;
        Block(void *b, void *p, size_t s, bool f)
            : base(b), ptr(p), size(s), isFree(f) {}
        bool operator<(const Block &other) const {
            return ptr < other.ptr;
        }
    };

    void *allocateNewRegion(size_t size);
    void insertFreeBlock(Block &&block);
    void tryCoalesce(const Block &block);

    std::vector<void *> baseRegions;
    std::set<Block> allBlocks;
    std::multimap<size_t, std::set<Block>::iterator> freeBlocks;
    std::unordered_map<void *, std::set<Block>::iterator> ptrToBlock;
};

class WorkspaceAllocator : public AllocatorBase {
private:
    void *_memory;
    size_t _total_size;
    size_t _align;

public:
    WorkspaceAllocator(size_t intial_size, size_t align = 256);
    ~WorkspaceAllocator();
    void *alloc(size_t size) override;
    void release(void *ptr) override;
};

#endif
