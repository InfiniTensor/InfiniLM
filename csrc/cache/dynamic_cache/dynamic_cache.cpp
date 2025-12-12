#include "dynamic_cache.hpp"

namespace infinilm::cache {

// KVCacheLayer Implementation

KVCacheLayer::KVCacheLayer()
    : max_capacity(0),
      initial_capacity(4096),
      initial_batch_size(1),
      growth_factor(2.0f),
      initialized(false) {}

void KVCacheLayer::ensure_capacity(size_t batch_size, size_t num_kv_heads, size_t head_dim,
                                   size_t seq_len, infinicore::DataType dtype,
                                   const infinicore::Device &device, const CacheConfig &cache_config) {
    size_t required_capacity = seq_len + std::accumulate(cache_positions.begin(), cache_positions.end(), size_t(0), [](size_t a, size_t b) { return std::max(a, b); });

    // VALIDATION: Verify input parameters
    if (num_kv_heads == 0 || head_dim == 0 || seq_len == 0) {
        SPDLOG_ERROR("KVCacheLayer::ensure_capacity: Invalid parameters - num_kv_heads: {}, head_dim: {}, seq_len: {}",
                     num_kv_heads, head_dim, seq_len);
        throw std::runtime_error("KV cache ensure_capacity: invalid parameters");
    }

    // Store config parameters on first initialization
    if (!initialized) {
        initial_capacity = cache_config.initial_capacity;
        initial_batch_size = cache_config.initial_batch_size;
        growth_factor = cache_config.growth_factor;
    }

    // Lazy initialization
    if (!initialized) {
        // Use max of required capacity and initial capacity from config
        max_capacity = std::max(required_capacity, initial_capacity);

        // Use max of current batch size and initial batch size from config
        size_t alloc_batch_size = std::max(batch_size, initial_batch_size);

        k_cache = infinicore::Tensor::empty({alloc_batch_size, num_kv_heads, max_capacity, head_dim},
                                            dtype, device);
        v_cache = infinicore::Tensor::empty({alloc_batch_size, num_kv_heads, max_capacity, head_dim},
                                            dtype, device);
        cache_positions = std::vector<size_t>(alloc_batch_size, 0);
        initialized = true;

        spdlog::debug("Initialized KV cache with batch_size={}, capacity={} (config: initial_batch={}, initial_capacity={})",
                      alloc_batch_size, max_capacity, initial_batch_size, initial_capacity);

        // VALIDATION: Verify cache was created correctly
        if (k_cache->shape()[0] != alloc_batch_size || k_cache->shape()[1] != num_kv_heads || k_cache->shape()[2] != max_capacity || k_cache->shape()[3] != head_dim) {
            SPDLOG_ERROR("KVCacheLayer::ensure_capacity: Cache shape mismatch after initialization");
            throw std::runtime_error("KV cache initialization: shape mismatch");
        }
    }
    // Grow cache if needed using growth factor from config
    else if (required_capacity > max_capacity) {
        if (!cache_config.allow_expand) {
            SPDLOG_ERROR("KVCacheLayer::ensure_capacity: Cache expansion not allowed by config");
            throw std::runtime_error("KV cache expansion not allowed");
        }
        // Calculate new capacity using growth factor
        size_t new_capacity = static_cast<size_t>(
            std::max(static_cast<float>(max_capacity) * growth_factor,
                     static_cast<float>(required_capacity + max_capacity)));

        // Ensure we don't exceed max_position_embeddings if specified
        if (cache_config.max_kv_cache_length != 0) {
            new_capacity = std::min(new_capacity, cache_config.max_kv_cache_length);
        }

        // Ensure we grow by at least some minimum amount
        size_t min_growth = 256;
        if (new_capacity - max_capacity < min_growth) {
            new_capacity = max_capacity + min_growth;
        }

        size_t new_batch_size = std::max(batch_size, k_cache->shape()[0]);
        if (num_kv_heads != k_cache->shape()[1] || head_dim != k_cache->shape()[3]) {
            throw std::runtime_error("KVCache ensure_capacity: num_kv_heads or head_dim mismatch with existing cache.");
        }
        if (new_batch_size > cache_positions.size()) {
            cache_positions.resize(new_batch_size, 0);
        }

        auto k_new = infinicore::Tensor::empty({new_batch_size, num_kv_heads, new_capacity, head_dim},
                                               dtype, device);
        auto v_new = infinicore::Tensor::empty({new_batch_size, num_kv_heads, new_capacity, head_dim},
                                               dtype, device);

        spdlog::debug("Growing KV cache from capacity {} to {} (growth_factor={})",
                      max_capacity, new_capacity, growth_factor);

        // Copy existing cache data
        for (size_t b = 0; b < new_batch_size; ++b) {
            size_t cache_position = cache_positions[b];
            if (cache_position > 0) {
                auto k_slice = k_cache->narrow({{0, b, 1}, {2, 0, cache_position}});
                auto v_slice = v_cache->narrow({{0, b, 1}, {2, 0, cache_position}});
                k_new->narrow({{0, b, 1}, {2, 0, cache_position}})->copy_from(k_slice);
                v_new->narrow({{0, b, 1}, {2, 0, cache_position}})->copy_from(v_slice);
            }
        }

        k_cache = k_new;
        v_cache = v_new;
        max_capacity = new_capacity;

        // VALIDATION: Verify cache was grown correctly
        if (k_cache->shape()[2] != new_capacity) {
            SPDLOG_ERROR("KVCacheLayer::ensure_capacity: New cache capacity mismatch");
            throw std::runtime_error("KV cache growth: capacity mismatch");
        }
    }

    // VALIDATION: Final check that capacity is sufficient
    if (required_capacity > max_capacity) {
        SPDLOG_ERROR("KVCacheLayer::ensure_capacity: Capacity still insufficient after growth");
        throw std::runtime_error("KV cache ensure_capacity: capacity insufficient");
    }
}

std::pair<infinicore::Tensor, infinicore::Tensor> KVCacheLayer::update(
    const infinicore::Tensor &k_new,
    const infinicore::Tensor &v_new,
    const CacheConfig &cache_config) {
    if (k_new->ndim() != 4 || v_new->ndim() != 4) {
        throw std::runtime_error("KVCache update: k_new and v_new must be 4D tensors");
    }
    size_t batch_size = k_new->shape()[0];
    size_t num_kv_heads = k_new->shape()[1];
    size_t seq_len = k_new->shape()[2];
    size_t head_dim = k_new->shape()[3];

    // Ensure capacity with cache config
    ensure_capacity(batch_size, num_kv_heads, head_dim, seq_len,
                    k_new->dtype(), k_new->device(), cache_config);

    // Copy new k/v into cache at current position
    bool all_equal = cache_positions.empty() || std::equal(cache_positions.begin() + 1, cache_positions.end(), cache_positions.begin());
    if (all_equal) {
        auto cache_position = cache_positions[0];

        auto k_dst = k_cache->narrow({{2, cache_position, seq_len}});
        auto v_dst = v_cache->narrow({{2, cache_position, seq_len}});
        k_dst->copy_from(k_new);
        v_dst->copy_from(v_new);

        // Update position
        cache_position += seq_len;
        for (size_t b = 0; b < batch_size; ++b) {
            cache_positions[b] = cache_position;
        }

        // Return the total cache up to current position
        auto k_total = k_cache->narrow({{2, 0, cache_position}});
        auto v_total = v_cache->narrow({{2, 0, cache_position}});

        return std::make_pair(k_total, v_total);
    } else {
        throw std::runtime_error("KVCache update: cache positions must be equal among a batch.");
    }
}

// DynamicCache Implementation

DynamicCache::DynamicCache(const CacheConfig &cache_config)
    : cache_config_(cache_config), layers_(cache_config.num_layers) {
    if (cache_config.num_layers == 0) {
        throw std::runtime_error("DynamicCache: num_layers must be specified in CacheConfig");
    }
}

DynamicCache::DynamicCache(size_t num_layers, size_t max_position_embeddings)
    : cache_config_(CacheConfig(CacheType::DYNAMIC, num_layers, max_position_embeddings)),
      layers_(num_layers) {
    if (num_layers == 0) {
        throw std::runtime_error("DynamicCache: num_layers must be greater than 0");
    }
}

std::pair<infinicore::Tensor, infinicore::Tensor> DynamicCache::update(
    size_t layer_idx,
    const infinicore::Tensor &k_new,
    const infinicore::Tensor &v_new) {
    if (layer_idx >= layers_.size()) {
        SPDLOG_ERROR("DynamicCache::update: layer_idx {} out of range (num_layers: {})",
                     layer_idx, layers_.size());
        throw std::runtime_error("DynamicCache: layer_idx out of range");
    }

    // Update the cache for this layer with cache config
    return layers_[layer_idx].update(k_new, v_new, cache_config_);
}

std::pair<infinicore::Tensor, infinicore::Tensor> DynamicCache::update(
    const infinicore::Tensor &k_new,
    const infinicore::Tensor &v_new) {
    return update(0, k_new, v_new);
}

const CacheConfig &DynamicCache::get_config() const {
    return cache_config_;
}

void DynamicCache::update_config(const CacheConfig &new_config) {
    // Check if we need to rebuild
    bool need_rebuild = false;

    // Rebuild if number of layers changed
    if (new_config.num_layers != cache_config_.num_layers || new_config.initial_batch_size != cache_config_.initial_batch_size) {
        need_rebuild = true;
        layers_.resize(new_config.num_layers);
    }

    // Rebuild if reset mode is RECREATE
    if (new_config.reset_mode == CacheResetMode::RECREATE) {
        need_rebuild = true;
    }

    // Update configuration
    cache_config_ = new_config;

    if (need_rebuild) {
        // Clear all layers to force reinitialization on next use
        for (auto &layer : layers_) {
            layer.initialized = false;
            layer.max_capacity = 0;
            // Tensors will be recreated when ensure_capacity is called
        }
        spdlog::info("DynamicCache configuration updated - cache will be rebuilt on next use");
    } else {
        spdlog::info("DynamicCache configuration updated: layers={}, initial_capacity={}, growth_factor={}",
                     new_config.num_layers, new_config.initial_capacity, new_config.growth_factor);
    }
}

size_t DynamicCache::num_layers() const {
    return layers_.size();
}

size_t DynamicCache::cache_position(size_t layer_idx) const {
    if (layer_idx >= layers_.size()) {
        throw std::runtime_error("DynamicCache: layer_idx out of range");
    }
    if (layers_[layer_idx].cache_positions.empty()) {
        return 0;
    }
    return layers_[layer_idx].cache_positions[0];
}

bool DynamicCache::is_initialized() const {
    return !layers_.empty() && layers_[0].initialized;
}

size_t DynamicCache::max_kv_cache_length() const {
    return cache_config_.max_kv_cache_length;
}

void DynamicCache::reset(size_t pos) {
    for (auto &layer : layers_) {
        std::fill(layer.cache_positions.begin(), layer.cache_positions.end(), pos);
        // Note: We don't reset initialized flag or clear the cache tensors
        // to avoid reallocation. The cache will be overwritten on next update.
    }
}

KVCacheLayer &DynamicCache::layer(size_t layer_idx) {
    if (layer_idx >= layers_.size()) {
        throw std::runtime_error("DynamicCache: layer_idx out of range");
    }
    return layers_[layer_idx];
}

const KVCacheLayer &DynamicCache::layer(size_t layer_idx) const {
    if (layer_idx >= layers_.size()) {
        throw std::runtime_error("DynamicCache: layer_idx out of range");
    }
    return layers_[layer_idx];
}

} // namespace infinilm::cache
