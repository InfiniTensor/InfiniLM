// cache_manager.cpp
#include "cache_manager.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::cache {

// Concrete implementation of CacheInterface for DynamicCache
class DynamicCacheWrapper : public CacheInterface {
public:
    DynamicCacheWrapper(const CacheConfig &cache_config)
        : cache_config_(cache_config),
          cache_(std::make_shared<DynamicCache>(cache_config)) {}

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

    // Get cache configuration
    const CacheConfig &get_config() const { return cache_config_; }

    // Update cache configuration
    void update_config(const CacheConfig &new_config) {
        cache_config_ = new_config;
        cache_->update_config(new_config);
    }

private:
    CacheConfig cache_config_;
    std::shared_ptr<DynamicCache> cache_;
};

// CacheManager Implementation
CacheManager::CacheManager(size_t num_caches,
                           const CacheConfig &cache_config) // Change parameter to CacheConfig
    : cache_config_(cache_config) {

    spdlog::info("Creating CacheManager with {} caches", num_caches);
    spdlog::info("Cache config: type={}, layers={}, initial_capacity={}, growth_factor={}, initial_batch={}",
                 static_cast<int>(cache_config.type),
                 cache_config.num_layers,
                 cache_config.initial_capacity,
                 cache_config.growth_factor,
                 cache_config.initial_batch_size);

    for (size_t i = 0; i < num_caches; ++i) {
        caches_.push_back(create_cache_instance());
    }
}

std::shared_ptr<CacheInterface> CacheManager::create_cache_instance() {
    switch (cache_config_.type) {
    case CacheType::DYNAMIC:
        return std::make_shared<DynamicCacheWrapper>(cache_config_);
    default:
        throw std::runtime_error("Unsupported cache type");
    }
}

// Add method to update cache configuration
bool CacheManager::reconfigure(const CacheConfig &new_config) {
    std::lock_guard<std::mutex> lock(mutex_);

    // Check if anything actually changed
    if (new_config == cache_config_) {
        return false; // No change needed
    }

    spdlog::info("Reconfiguring cache: type={}->{}, layers={}->{}, initial_capacity={}->{}, growth_factor={}->{}",
                 static_cast<int>(cache_config_.type), static_cast<int>(new_config.type),
                 cache_config_.num_layers, new_config.num_layers,
                 cache_config_.initial_capacity, new_config.initial_capacity,
                 cache_config_.growth_factor, new_config.growth_factor);

    // Update configuration
    cache_config_ = new_config;

    // Clear existing caches
    caches_.clear();

    // Create new caches with updated configuration
    size_t num_caches = caches_.size(); // This should be preserved from before
    for (size_t i = 0; i < num_caches; ++i) {
        caches_.push_back(create_cache_instance());
    }

    return true;
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
