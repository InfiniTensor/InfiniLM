#ifndef CACHE_MANAGER_HPP
#define CACHE_MANAGER_HPP

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../tensor.hpp"
#include "../utils.hpp"
#include "infinicore_infer.h"

class IDescriptorDestroyer {
public:
    virtual ~IDescriptorDestroyer() = default;
    virtual void destroy(void *descriptor) = 0;
};

template <typename DescriptorType>
class DescriptorDestroyer : public IDescriptorDestroyer {
    using DestroyFunc = infiniStatus_t (*)(DescriptorType);
    DestroyFunc destroyFunc;

public:
    DescriptorDestroyer(DestroyFunc func) : destroyFunc(func) {}

    void destroy(void *descriptor) override {
        destroyFunc(*static_cast<DescriptorType *>(descriptor));
    }
};

template <typename DescriptorType>
class LRUDescriptorCache {
private:
    struct CacheNode {
        size_t key;
        DescriptorType desc;
        CacheNode *prev;
        CacheNode *next;

        CacheNode() : key(0), desc(), prev(nullptr), next(nullptr) {}
        CacheNode(size_t k, const DescriptorType &d) : key(k), desc(d), prev(nullptr), next(nullptr) {}
    };

    std::unordered_map<size_t, CacheNode *> cache;
    CacheNode *head;
    CacheNode *tail;
    const size_t capacity;
    size_t size;
    std::unique_ptr<IDescriptorDestroyer> destroyer;

    void removeNode(CacheNode *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        if (destroyer) {
            destroyer->destroy(&node->desc);
        }
        cache.erase(node->key);
        delete node;
        --size;
    }

    void addToTop(CacheNode *node) {
        node->next = head->next;
        node->next->prev = node;
        node->prev = head;
        head->next = node;
        cache[node->key] = node;
        if (++size > capacity) {
            removeNode(tail->prev);
        }
    }

    void moveToTop(CacheNode *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        node->next = head->next;
        node->next->prev = node;
        node->prev = head;
        head->next = node;
    }

public:
    template <typename DestroyFunc>
    LRUDescriptorCache(size_t c, DestroyFunc destroyFunc)
        : capacity(c), size(0), destroyer(std::make_unique<DescriptorDestroyer<DescriptorType>>(destroyFunc)) {
        head = new CacheNode();
        tail = new CacheNode();
        head->next = tail;
        tail->prev = head;
    }

    ~LRUDescriptorCache() {
        while (head->next != tail) {
            removeNode(head->next);
        }
        delete head;
        delete tail;
    }

    bool get(size_t key, DescriptorType &out_desc) {
        auto it = cache.find(key);
        if (it == cache.end()) {
            return false;
        }

        CacheNode *node = it->second;
        moveToTop(node);
        out_desc = node->desc;
        return true;
    }

    void put(size_t key, const DescriptorType &descriptor) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            // Key already exists, update the descriptor
            CacheNode *node = it->second;
            if (destroyer) {
                destroyer->destroy(&node->desc);
            }
            node->desc = descriptor;
            moveToTop(node);
            return;
        }

        // Check if we need to evict
        if (size >= capacity) {
            removeNode(tail->prev);
        }

        // Create new node and add to top
        CacheNode *node = new CacheNode(key, descriptor);
        addToTop(node);
    }

    LRUDescriptorCache(const LRUDescriptorCache &) = delete;
    LRUDescriptorCache &operator=(const LRUDescriptorCache &) = delete;
};

// Helper macro to generate the destroy function name
#define DESTROY_FUNC(OpType) infiniopDestroy##OpType##Descriptor

// 宏展开后的结果
// LRUDescriptorCache<infiniopAddDescriptor_t> Add_cache;

// bool getAddDescriptor(size_t key, infiniopAddDescriptor_t &desc) {
//     return Add_cache.get(key, desc);
// }

// void putAddDescriptor(size_t key, const infiniopAddDescriptor_t &desc) {
//     Add_cache.put(key, desc);
// }

// Declare cache and access functions
#define DECLARE_OP_CACHE(OpType)                                                           \
    LRUDescriptorCache<infiniop##OpType##Descriptor_t> OpType##_cache;                     \
    bool get##OpType##Descriptor(size_t key, infiniop##OpType##Descriptor_t &desc) {       \
        return OpType##_cache.get(key, desc);                                              \
    }                                                                                      \
    void put##OpType##Descriptor(size_t key, const infiniop##OpType##Descriptor_t &desc) { \
        OpType##_cache.put(key, desc);                                                     \
    }

class CacheManager {
public:
    DECLARE_OP_CACHE(Add)
    DECLARE_OP_CACHE(RMSNorm)
    DECLARE_OP_CACHE(Gemm)
    DECLARE_OP_CACHE(RoPE)
    DECLARE_OP_CACHE(Rearrange)
    DECLARE_OP_CACHE(CausalSoftmax)
    DECLARE_OP_CACHE(Topkrouter)
    DECLARE_OP_CACHE(SwiGLU)
    DECLARE_OP_CACHE(RandomSample)
    DECLARE_OP_CACHE(DequantizeAWQ)
    DECLARE_OP_CACHE(Softmax)  // 新增  
    DECLARE_OP_CACHE(BiAttention)


    CacheManager(size_t capacity = 100)
        : Add_cache(capacity, DESTROY_FUNC(Add)),
          RMSNorm_cache(capacity, DESTROY_FUNC(RMSNorm)),
          Gemm_cache(capacity, DESTROY_FUNC(Gemm)),
          RoPE_cache(capacity, DESTROY_FUNC(RoPE)),
          Rearrange_cache(capacity, DESTROY_FUNC(Rearrange)),
          CausalSoftmax_cache(capacity, DESTROY_FUNC(CausalSoftmax)),
          Topkrouter_cache(capacity, DESTROY_FUNC(Topkrouter)),
          SwiGLU_cache(capacity, DESTROY_FUNC(SwiGLU)),
          RandomSample_cache(capacity, DESTROY_FUNC(RandomSample)),
          DequantizeAWQ_cache(capacity, DESTROY_FUNC(DequantizeAWQ)),
          Softmax_cache(capacity, DESTROY_FUNC(Softmax)),
          BiAttention_cache(capacity, DESTROY_FUNC(BiAttention)){}

    template <typename... Tensors>
    static size_t createDescriptorKey(Tensors... tensors) {
        size_t seed = 0;
        (..., (tensors ? hash_combine(seed, tensors->seed()) : (void)0));
        return seed;
    }
};

#undef DESTROY_FUNC
#undef DECLARE_OP_CACHE

#endif // CACHE_MANAGER_HPP
