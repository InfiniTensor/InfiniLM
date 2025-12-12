#pragma once

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"

#include "../cache_config.hpp"
#include "../cache_interface.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace infinilm::cache {

/**
 * @brief Single layer's KV cache for incremental decoding
 *
 * Stores key and value caches with shape [batch_size, n_kv_head, capacity, head_dim]
 * Similar to DynamicLayer in Python cache_utils.py
 *
 * This represents a single layer's cache within a model-level cache container.
 */
struct KVCacheLayer {
    infinicore::Tensor k_cache;          // [batch_size, n_kv_head, capacity, head_dim]
    infinicore::Tensor v_cache;          // [batch_size, n_kv_head, capacity, head_dim]
    std::vector<size_t> cache_positions; // Current position in cache
    size_t max_capacity;                 // Maximum capacity of cache
    size_t initial_capacity;             // Initial capacity from config
    size_t initial_batch_size;           // Initial batch size from config
    float growth_factor;                 // Growth factor for dynamic resizing
    bool initialized;                    // Whether cache has been initialized

    /**
     * @brief Default constructor
     */
    KVCacheLayer();

    /**
     * @brief Initialize or update cache capacity with config parameters
     * @param batch_size Current batch size
     * @param num_kv_heads Number of key-value heads
     * @param head_dim Head dimension
     * @param seq_len Sequence length of new tokens
     * @param dtype Data type
     * @param device Device
     * @param cache_config Cache configuration parameters
     */
    void ensure_capacity(size_t batch_size, size_t num_kv_heads, size_t head_dim, size_t seq_len,
                         infinicore::DataType dtype, const infinicore::Device &device,
                         const CacheConfig &cache_config);

    /**
     * @brief Update cache with new key and value states
     * @param k_new New key states [batch_size, n_kv_head, seq_len, head_dim]
     * @param v_new New value states [batch_size, n_kv_head, seq_len, head_dim]
     * @param cache_config Cache configuration for capacity management
     * @return Tuple of (k_total, v_total) with shape [batch_size, n_kv_head, total_seq_len, head_dim]
     */
    std::pair<infinicore::Tensor, infinicore::Tensor> update(
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new,
        const CacheConfig &cache_config);
};

/**
 * @brief Model-level KV cache container (similar to DynamicCache in Python)
 *
 * Stores a list of KVCacheLayer objects, one per model layer.
 * This aligns with Python backend's DynamicCache architecture.
 */
class DynamicCache : public Cache {
public:
    /**
     * @brief Construct DynamicCache with cache configuration
     * @param cache_config Cache configuration parameters
     */
    explicit DynamicCache(const CacheConfig &cache_config);

    /**
     * @brief Construct DynamicCache with specified number of layers
     *
     * @param num_layers Number of model layers (creates one cache layer per model layer)
     * @param max_position_embeddings Maximum position embeddings (used for initial capacity)
     */
    DynamicCache(size_t num_layers, size_t max_position_embeddings = 4096);

    /**
     * @brief Update cache with new key and value states for a specific layer
     */
    std::pair<infinicore::Tensor, infinicore::Tensor> update(
        size_t layer_idx,
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new) override;

    /**
     * @brief Update cache with new key and value states (convenience method without layer_idx)
     * This is used when the cache is accessed directly without layer information
     *
     * @param k_new New key states [batch_size, n_kv_head, seq_len, head_dim]
     * @param v_new New value states [batch_size, n_kv_head, seq_len, head_dim]
     * @return Tuple of (k_total, v_total) with shape [batch_size, n_kv_head, total_seq_len, head_dim]
     *
     * Note: This assumes layer_idx=0. For multi-layer models, use update(layer_idx, k_new, v_new) instead.
     */
    std::pair<infinicore::Tensor, infinicore::Tensor> update(
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new) override;

    /**
     * @brief Get cache configuration
     */
    const CacheConfig &get_config() const override;

    /**
     * @brief Update cache configuration (for dynamic reconfiguration)
     */
    void update_config(const CacheConfig &new_config) override;

    /**
     * @brief Get the number of layers in this cache
     */
    size_t num_layers() const override;

    /**
     * @brief Get cache position for a specific layer
     */
    size_t cache_position(size_t layer_idx) const override;

    /**
     * @brief Check if cache is initialized
     */
    bool is_initialized() const override;

    /**
     * @brief Get max position embeddings (used for initial capacity)
     */
    size_t max_kv_cache_length() const;

    /**
     * @brief Reset cache for all layers to a specific position
     * This should be called when starting a new generation sequence or resetting to a specific position
     * @param pos Position to reset to (defaults to 0)
     */
    void reset(size_t pos = 0) override;

    /**
     * @brief Access a specific layer's cache (for advanced usage)
     */
    KVCacheLayer &layer(size_t layer_idx);

    /**
     * @brief Access a specific layer's cache (const version)
     */
    const KVCacheLayer &layer(size_t layer_idx) const;

private:
    CacheConfig cache_config_;
    std::vector<KVCacheLayer> layers_;
};

} // namespace infinilm::cache
