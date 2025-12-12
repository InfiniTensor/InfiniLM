#pragma once

#include "cache_config.hpp"
#include "infinicore/tensor.hpp"

#include <memory>

namespace infinilm::cache {

/**
 * @brief Abstract interface for KV cache implementations
 * This allows different cache types (Dynamic, Paged, etc.) to be used interchangeably
 */
class Cache {
public:
    virtual ~Cache() = default;

    /**
     * @brief Update cache with new key and value states
     * @param layer_idx Layer index for multi-layer models
     * @param k_new New key states [batch_size, n_kv_head, seq_len, head_dim]
     * @param v_new New value states [batch_size, n_kv_head, seq_len, head_dim]
     * @return Tuple of (k_total, v_total) with shape [batch_size, n_kv_head, total_seq_len, head_dim]
     */
    virtual std::pair<infinicore::Tensor, infinicore::Tensor> update(
        size_t layer_idx,
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new)
        = 0;

    /**
     * @brief Update cache (convenience method for single-layer or default layer)
     */
    virtual std::pair<infinicore::Tensor, infinicore::Tensor> update(
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new) {
        return update(0, k_new, v_new);
    }

    /**
     * @brief Reset cache for all layers to a specific position
     * @param pos Position to reset to (defaults to 0)
     */
    virtual void reset(size_t pos = 0) = 0;

    /**
     * @brief Update cache configuration
     * @param new_config New cache configuration
     */
    virtual void update_config(const CacheConfig &new_config) = 0;

    /**
     * @brief Get current cache configuration
     */
    virtual const CacheConfig &get_config() const = 0;

    /**
     * @brief Get the number of layers in this cache
     */
    virtual size_t num_layers() const = 0;

    /**
     * @brief Get cache position for a specific layer
     */
    virtual size_t cache_position(size_t layer_idx) const = 0;

    /**
     * @brief Check if cache is initialized
     */
    virtual bool is_initialized() const = 0;

    /**
     * @brief Factory method to create cache based on configuration
     */
    static std::shared_ptr<Cache> create(const CacheConfig &config);
};

} // namespace infinilm::cache
