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
    static constexpr size_t DEFAULT_ALIGNMENT = 256;

    explicit MemoryPool(size_t initialSize = 0, size_t alignment = DEFAULT_ALIGNMENT);
    ~MemoryPool();

    void *alloc(size_t size) override;
    void release(void *ptr) override;

    size_t getAlignment() const { return _alignment; }

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
    void tryCoalesce(const Block &block);

    size_t _alignment;
    std::vector<void *> _base_regions;
    std::set<Block> _all_blocks;
    std::multimap<size_t, std::set<Block>::iterator> _free_blocks;
    std::unordered_map<void *, std::set<Block>::iterator> _ptr_to_block;
};

#endif
