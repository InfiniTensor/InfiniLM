#pragma once

#include "cache_config.hpp"
#include "infinicore/device.hpp"
#include "kv_cache.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace infinilm::cache {

/**
 * @class CacheManager
 * @brief Manages multiple cache instances across devices/workers
 *
 * In distributed settings, each device/worker gets its own cache instance
 * from the manager's vector. Each decoder layer corresponds to a KVCacheLayer
 * within a DynamicCache (or equivalent in other cache types).
 */
class CacheManager {
public:
    /**
     * @brief Construct CacheManager with cache configuration
     * @param num_caches Number of cache instances to create
     * @param cache_config Cache configuration
     */
    CacheManager(size_t num_caches = 1,
                 const CacheConfig &cache_config = CacheConfig());

    /**
     * @brief Reconfigure cache with new configuration
     * @param new_config New cache configuration
     * @return True if cache was reconfigured, false if parameters unchanged
     */
    bool reconfigure(const CacheConfig &new_config);

    /**
     * @brief Get current cache configuration
     */
    const CacheConfig &get_cache_config() const { return cache_config_; }

    /**
     * @brief Get cache type
     */
    CacheType cache_type() const { return cache_config_.type; }

    /**
     * @brief Get the number of layers
     */
    size_t get_num_layers() const { return cache_config_.num_layers; }

    /**
     * @brief Get max KV cache length
     */
    size_t get_max_kv_cache_length() const { return cache_config_.max_kv_cache_length; }

    /**
     * @brief Get cache for a specific worker/device
     * @param worker_idx Worker/device index (0-based)
     * @return Reference to cache interface for the worker
     */
    CacheInterface &get_cache(size_t worker_idx);

    /**
     * @brief Get cache for a specific worker/device (const version)
     */
    const CacheInterface &get_cache(size_t worker_idx) const;

    /**
     * @brief Get raw pointer to cache for a specific worker/device
     * @param worker_idx Worker/device index
     * @return void* pointer suitable for passing to forward() methods
     */
    void *get_raw_cache_ptr(size_t worker_idx);

    /**
     * @brief Update cache for a specific worker and layer
     * @param worker_idx Worker index
     * @param layer_idx Layer index within the model
     * @param k_new New key states [batch, n_kv_head, seq_len, head_dim]
     * @param v_new New value states [batch, n_kv_head, seq_len, head_dim]
     */
    void update_cache(size_t worker_idx,
                      size_t layer_idx,
                      const infinicore::Tensor &k_new,
                      const infinicore::Tensor &v_new);

    /**
     * @brief Reset cache for a specific worker
     * @param worker_idx Worker index
     * @param pos Position to reset to (default 0)
     */
    void reset_worker(size_t worker_idx, size_t pos = 0);

    /**
     * @brief Get current cache position for a worker and layer
     */
    size_t get_cache_position(size_t worker_idx, size_t layer_idx) const;

    /**
     * @brief Get number of cache instances
     */
    size_t num_caches() const { return caches_.size(); }

    /**
     * @brief Create a new cache instance (for dynamic scaling)
     * @return Index of the newly created cache
     */
    size_t create_new_cache();

    /**
     * @brief Remove a cache instance
     * @param worker_idx Index of cache to remove
     */
    void remove_cache(size_t worker_idx);

    /**
     * @brief Reconfigure cache with new parameters
     * @param new_type New cache type
     * @param new_num_layers New number of layers
     * @param new_max_position_embeddings New max position embeddings
     * @return True if cache was reconfigured, false if parameters unchanged
     */
    bool reconfigure(CacheType new_type,
                     size_t new_num_layers,
                     size_t new_max_position_embeddings);

    /**
     * @brief Get total memory usage across all caches
     */
    size_t total_memory_usage() const;

    /**
     * @brief Clear all caches (free memory)
     */
    void clear_all();

private:
    std::vector<std::shared_ptr<CacheInterface>> caches_;
    CacheConfig cache_config_;

    // Factory method to create cache instances
    std::shared_ptr<CacheInterface> create_cache_instance();
};

} // namespace infinilm::cache
