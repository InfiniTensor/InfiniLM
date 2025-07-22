#ifndef CACHE_MANAGER_HPP
#define CACHE_MANAGER_HPP

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../tensor.hpp"
#include "../utils.hpp"
#include "infinicore_infer.h"

// Hash combine utility (similar to boost::hash_combine)
inline void hash_combine(size_t &seed, size_t value) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Specialization for enum types
template <typename T>
inline void hash_combine(size_t &seed, T value, typename std::enable_if<std::is_enum<T>::value>::type * = 0) {
    hash_combine(seed, static_cast<size_t>(value));
}

// Helper function to compute hash for tensor descriptors
inline size_t computeTensorDescHash(std::shared_ptr<Tensor> tensor) {
    size_t seed = 0;
    hash_combine(seed, tensor->dtype());
    for (auto dim : tensor->shape()) {
        hash_combine(seed, dim);
    }
    for (auto stride : tensor->strides()) {
        hash_combine(seed, static_cast<size_t>(stride));
    }
    return seed;
}

enum class OperatorType {
    RMS_NORM,
    GEMM,
    ROPE,
    REARRANGE,
    CAUSAL_SOFTMAX,
    SWIGLU,
    RANDOM_SAMPLE
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
    const OperatorType opType;

    void destroyDescriptor(DescriptorType &desc) {
        switch (opType) {
        case OperatorType::RMS_NORM:
            infiniopDestroyRMSNormDescriptor(desc);
            break;
        case OperatorType::GEMM:
            infiniopDestroyGemmDescriptor(desc);
            break;
        case OperatorType::ROPE:
            infiniopDestroyRoPEDescriptor(desc);
            break;
        case OperatorType::REARRANGE:
            infiniopDestroyRearrangeDescriptor(desc);
            break;
        case OperatorType::CAUSAL_SOFTMAX:
            infiniopDestroyCausalSoftmaxDescriptor(desc);
            break;
        case OperatorType::SWIGLU:
            infiniopDestroySwiGLUDescriptor(desc);
            break;
        case OperatorType::RANDOM_SAMPLE:
            infiniopDestroyRandomSampleDescriptor(desc);
            break;
        default:
            throw std::runtime_error("Unknown descriptor type");
        }
    }

    void removeNode(CacheNode *node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
        destroyDescriptor(node->desc);
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
    LRUDescriptorCache(size_t c, OperatorType t) : capacity(c), size(0), opType(t) {
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
            destroyDescriptor(node->desc);
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

class CacheManager {
private:
    const size_t DEFAULT_CACHE_CAPACITY = 128;

    LRUDescriptorCache<infiniopRMSNormDescriptor_t> rms_norm_cache;
    LRUDescriptorCache<infiniopGemmDescriptor_t> gemm_cache;
    LRUDescriptorCache<infiniopRoPEDescriptor_t> rope_cache;
    LRUDescriptorCache<infiniopRearrangeDescriptor_t> rearrange_cache;
    LRUDescriptorCache<infiniopCausalSoftmaxDescriptor_t> causal_softmax_cache;
    LRUDescriptorCache<infiniopSwiGLUDescriptor_t> swiglu_cache;
    LRUDescriptorCache<infiniopRandomSampleDescriptor_t> random_sample_cache;

public:
    CacheManager(size_t capacity = 100) : rms_norm_cache(capacity, OperatorType::RMS_NORM),
                                          gemm_cache(capacity, OperatorType::GEMM),
                                          rope_cache(capacity, OperatorType::ROPE),
                                          rearrange_cache(capacity, OperatorType::REARRANGE),
                                          causal_softmax_cache(capacity, OperatorType::CAUSAL_SOFTMAX),
                                          swiglu_cache(capacity, OperatorType::SWIGLU),
                                          random_sample_cache(capacity, OperatorType::RANDOM_SAMPLE) {}

    // RMSNorm operations
    bool getRMSNormDescriptor(size_t key, infiniopRMSNormDescriptor_t &desc) {
        return rms_norm_cache.get(key, desc);
    }

    void putRMSNormDescriptor(size_t key, const infiniopRMSNormDescriptor_t &desc) {
        rms_norm_cache.put(key, desc);
    }

    // GEMM operations
    bool getGemmDescriptor(size_t key, infiniopGemmDescriptor_t &desc) {
        return gemm_cache.get(key, desc);
    }

    void putGemmDescriptor(size_t key, const infiniopGemmDescriptor_t &desc) {
        gemm_cache.put(key, desc);
    }

    // RoPE operations
    bool getRoPEDescriptor(size_t key, infiniopRoPEDescriptor_t &desc) {
        return rope_cache.get(key, desc);
    }

    void putRoPEDescriptor(size_t key, const infiniopRoPEDescriptor_t &desc) {
        rope_cache.put(key, desc);
    }

    // Rearrange operations
    bool getRearrangeDescriptor(size_t key, infiniopRearrangeDescriptor_t &desc) {
        return rearrange_cache.get(key, desc);
    }

    void putRearrangeDescriptor(size_t key, const infiniopRearrangeDescriptor_t &desc) {
        rearrange_cache.put(key, desc);
    }

    // Softmax operations
    bool getCausalSoftmaxDescriptor(size_t key, infiniopCausalSoftmaxDescriptor_t &desc) {
        return causal_softmax_cache.get(key, desc);
    }

    void putCausalSoftmaxDescriptor(size_t key, const infiniopCausalSoftmaxDescriptor_t &desc) {
        causal_softmax_cache.put(key, desc);
    }

    // SwiGLU operations
    bool getSwiGLUDescriptor(size_t key, infiniopSwiGLUDescriptor_t &desc) {
        return swiglu_cache.get(key, desc);
    }

    void putSwiGLUDescriptor(size_t key, const infiniopSwiGLUDescriptor_t &desc) {
        swiglu_cache.put(key, desc);
    }

    // Random Sample operations
    bool getRandomSampleDescriptor(size_t key, infiniopRandomSampleDescriptor_t &desc) {
        return random_sample_cache.get(key, desc);
    }

    void putRandomSampleDescriptor(size_t key, const infiniopRandomSampleDescriptor_t &desc) {
        random_sample_cache.put(key, desc);
    }

    static size_t createDescriptorKey(std::shared_ptr<Tensor> desc0,
                                      std::shared_ptr<Tensor> desc1,
                                      std::shared_ptr<Tensor> desc2,
                                      std::shared_ptr<Tensor> desc3,
                                      std::shared_ptr<Tensor> desc4) {
        size_t seed = 0;
        if (desc0) {
            hash_combine(seed, computeTensorDescHash(desc0));
        }
        if (desc1) {
            hash_combine(seed, computeTensorDescHash(desc1));
        }
        if (desc2) {
            hash_combine(seed, computeTensorDescHash(desc2));
        }
        if (desc3) {
            hash_combine(seed, computeTensorDescHash(desc3));
        }
        if (desc4) {
            hash_combine(seed, computeTensorDescHash(desc4));
        }
        return seed;
    }
};

#endif // CACHE_MANAGER_HPP
