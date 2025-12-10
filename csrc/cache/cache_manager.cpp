// cache_manager.cpp
#include "cache_manager.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::cache {

// Concrete implementation of CacheInterface for DynamicCache
class DynamicCacheWrapper : public CacheInterface {
public:
    DynamicCacheWrapper(size_t num_layers, size_t max_position_embeddings)
        : cache_(std::make_unique<DynamicCache>(num_layers, max_position_embeddings)) {}

    void update(size_t layer_idx,
                const infinicore::Tensor &k_new,
                const infinicore::Tensor &v_new) override {
        cache_->update(layer_idx, k_new, v_new);
    }

    void reset(size_t pos = 0) override {
        cache_->reset(pos);
    }

    size_t cache_position(size_t layer_idx) const override {
        return cache_->cache_position(layer_idx);
    }

    size_t num_layers() const override {
        return cache_->num_layers();
    }

    void *raw_ptr() override {
        return cache_.get();
    }

    CacheType type() const override {
        return CacheType::DYNAMIC;
    }

    std::string name() const override {
        return "DynamicCache";
    }

    size_t memory_usage() const override {
        // Simple estimation: sum of all tensor sizes
        size_t total = 0;
        for (size_t i = 0; i < num_layers(); ++i) {
            const auto &layer = cache_->layer(i);
            if (layer.initialized) {
                total += layer.k_cache->numel() * layer.k_cache->element_size();
                total += layer.v_cache->numel() * layer.v_cache->element_size();
            }
        }
        return total;
    }

private:
    std::unique_ptr<DynamicCache> cache_;
};

// Placeholder for future cache types
class StaticCacheWrapper : public CacheInterface {
public:
    StaticCacheWrapper(size_t num_layers, size_t max_position_embeddings) {
        // TODO: Implement when StaticCache is available
        throw std::runtime_error("StaticCache not implemented yet");
    }

    void update(size_t layer_idx,
                const infinicore::Tensor &k_new,
                const infinicore::Tensor &v_new) override {
        // TODO: Implement
    }

    void reset(size_t pos = 0) override {
        // TODO: Implement
    }

    size_t cache_position(size_t layer_idx) const override {
        // TODO: Implement
        return 0;
    }

    size_t num_layers() const override {
        // TODO: Implement
        return 0;
    }

    void *raw_ptr() override {
        // TODO: Implement
        return nullptr;
    }

    CacheType type() const override {
        return CacheType::STATIC;
    }

    std::string name() const override {
        return "StaticCache";
    }

    bool supports_prefill() const override {
        return true;
    }

    bool supports_incremental() const override {
        return false; // Static cache might not support incremental decoding
    }
};

// Placeholder for FlashAttention cache
class FlashAttentionCacheWrapper : public CacheInterface {
public:
    FlashAttentionCacheWrapper(size_t num_layers, size_t max_position_embeddings) {
        // TODO: Implement when FlashAttentionCache is available
        throw std::runtime_error("FlashAttentionCache not implemented yet");
    }

    void update(size_t layer_idx,
                const infinicore::Tensor &k_new,
                const infinicore::Tensor &v_new) override {
        // TODO: Implement
    }

    void reset(size_t pos = 0) override {
        // TODO: Implement
    }

    size_t cache_position(size_t layer_idx) const override {
        // TODO: Implement
        return 0;
    }

    size_t num_layers() const override {
        // TODO: Implement
        return 0;
    }

    void *raw_ptr() override {
        // TODO: Implement
        return nullptr;
    }

    CacheType type() const override {
        return CacheType::FLASH_ATTN;
    }

    std::string name() const override {
        return "FlashAttentionCache";
    }

    bool supports_prefill() const override {
        return true;
    }

    bool supports_incremental() const override {
        return true;
    }
};

// CacheManager Implementation
CacheManager::CacheManager(size_t num_caches,
                           CacheType cache_type,
                           size_t num_layers,
                           size_t max_position_embeddings)
    : cache_type_(cache_type),
      num_layers_(num_layers),
      max_position_embeddings_(max_position_embeddings) {

    spdlog::info("Creating CacheManager with {} caches of type {}",
                 num_caches, static_cast<int>(cache_type));

    for (size_t i = 0; i < num_caches; ++i) {
        caches_.push_back(create_cache_instance());
    }
}

std::unique_ptr<CacheInterface> CacheManager::create_cache_instance() {
    switch (cache_type_) {
    case CacheType::DYNAMIC:
        return std::make_unique<DynamicCacheWrapper>(num_layers_, max_position_embeddings_);
    case CacheType::STATIC:
        return std::make_unique<StaticCacheWrapper>(num_layers_, max_position_embeddings_);
    case CacheType::FLASH_ATTN:
        return std::make_unique<FlashAttentionCacheWrapper>(num_layers_, max_position_embeddings_);
    default:
        throw std::runtime_error("Unsupported cache type");
    }
}

CacheInterface &CacheManager::get_cache(size_t worker_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_idx >= caches_.size()) {
        throw std::runtime_error("CacheManager: worker index out of range");
    }
    return *caches_[worker_idx];
}

const CacheInterface &CacheManager::get_cache(size_t worker_idx) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_idx >= caches_.size()) {
        throw std::runtime_error("CacheManager: worker index out of range");
    }
    return *caches_[worker_idx];
}

void *CacheManager::get_raw_cache_ptr(size_t worker_idx) {
    return get_cache(worker_idx).raw_ptr();
}

void CacheManager::update_cache(size_t worker_idx,
                                size_t layer_idx,
                                const infinicore::Tensor &k_new,
                                const infinicore::Tensor &v_new) {
    auto &cache = get_cache(worker_idx);
    cache.update(layer_idx, k_new, v_new);
}

void CacheManager::reset_all(size_t pos) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &cache : caches_) {
        cache->reset(pos);
    }
}

void CacheManager::reset_worker(size_t worker_idx, size_t pos) {
    get_cache(worker_idx).reset(pos);
}

size_t CacheManager::get_cache_position(size_t worker_idx, size_t layer_idx) const {
    return get_cache(worker_idx).cache_position(layer_idx);
}

size_t CacheManager::create_new_cache() {
    std::lock_guard<std::mutex> lock(mutex_);
    caches_.push_back(create_cache_instance());
    return caches_.size() - 1;
}

void CacheManager::remove_cache(size_t worker_idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (worker_idx >= caches_.size()) {
        throw std::runtime_error("CacheManager: worker index out of range");
    }
    caches_.erase(caches_.begin() + worker_idx);
}

size_t CacheManager::total_memory_usage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    size_t total = 0;
    for (const auto &cache : caches_) {
        total += cache->memory_usage();
    }
    return total;
}

void CacheManager::clear_all() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &cache : caches_) {
        cache->clear();
    }
}

} // namespace infinilm::cache
