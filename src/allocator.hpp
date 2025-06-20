#ifndef ALLOCATOR_HPP
#define ALLOCATOR_HPP

#include "infinicore_infer.h"
#include <map>
#include <memory>
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
        bool is_free;
        Block(void *b, void *p, size_t s, bool f)
            : base(b), ptr(p), size(s), is_free(f) {}
        bool operator<(const Block &other) const {
            return ptr < other.ptr;
        }
    };

    void *allocateNewRegion(size_t size);
    void insertFreeBlock(Block &&block);
    void tryCoalesce(const Block &block);

    std::vector<void *> _base_regions;
    std::set<Block> _all_blocks;
    std::multimap<size_t, std::set<Block>::iterator> _free_blocks;
    std::unordered_map<void *, std::set<Block>::iterator> _ptr_to_block;
};

#endif
