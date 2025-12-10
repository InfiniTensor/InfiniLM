#pragma once

#include "kv_cache.hpp"
#include "infinicore/device.hpp"
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace infinilm::cache {

// Forward declarations for future cache types
class StaticCache;
class FlashAttentionCache;

/**
 * @enum CacheType
 * @brief Enumeration of supported cache types
 */
enum class CacheType {
    DYNAMIC,   ///< Dynamic KV cache (grows as needed)
    STATIC,    ///< Static KV cache (fixed capacity)
    FLASH_ATTN ///< FlashAttention-optimized cache
};

/**
 * @class CacheInterface
 * @brief Abstract interface for all cache types
 */
class CacheInterface {
public:
    virtual ~CacheInterface() = default;

    // Common cache operations
    virtual void update(size_t layer_idx,
                        const infinicore::Tensor &k_new,
                        const infinicore::Tensor &v_new)
        = 0;

    virtual void reset(size_t pos = 0) = 0;
    virtual size_t cache_position(size_t layer_idx) const = 0;
    virtual size_t num_layers() const = 0;
    virtual void *raw_ptr() = 0; // Returns raw pointer for compatibility

    // Type information
    virtual CacheType type() const = 0;
    virtual std::string name() const = 0;

    // Optional advanced features
    virtual void clear() { reset(0); }
    virtual size_t memory_usage() const { return 0; }
    virtual bool supports_prefill() const { return true; }
    virtual bool supports_incremental() const { return true; }
};

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
     * @brief Construct CacheManager with specified configuration
     * @param num_caches Number of cache instances to create (typically equals number of workers)
     * @param cache_type Type of cache to create for all instances
     * @param num_layers Number of layers per cache (for DynamicCache)
     * @param max_position_embeddings Maximum position embeddings
     */
    CacheManager(size_t num_caches = 1,
                 CacheType cache_type = CacheType::DYNAMIC,
                 size_t num_layers = 32,
                 size_t max_position_embeddings = 4096);

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
     * @brief Reset cache for all workers
     * @param pos Position to reset to (default 0)
     */
    void reset_all(size_t pos = 0);

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
     * @brief Get cache type
     */
    CacheType cache_type() const { return cache_type_; }

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
     * @brief Get total memory usage across all caches
     */
    size_t total_memory_usage() const;

    /**
     * @brief Clear all caches (free memory)
     */
    void clear_all();

private:
    std::vector<std::unique_ptr<CacheInterface>> caches_;
    CacheType cache_type_;
    size_t num_layers_;
    size_t max_position_embeddings_;
    mutable std::mutex mutex_;

    // Factory method to create cache instances
    std::unique_ptr<CacheInterface> create_cache_instance();
};

} // namespace infinilm::cache
