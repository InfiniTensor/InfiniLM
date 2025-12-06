#pragma once

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"

#include "infinicore/context/context.hpp"
#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace infinilm::cache {

/**
 * @brief Helper function to format tensor shape as string for logging
 */
inline std::string shape_to_string(const std::vector<size_t>& shape) {
    if (shape.empty()) {
        return "[]";
    }
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) result += ", ";
        result += std::to_string(shape[i]);
    }
    result += "]";
    return result;
}

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
    bool initialized;                    // Whether cache has been initialized

    KVCacheLayer() : max_capacity(0), initialized(false) {}

    /**
     * @brief Initialize or update cache capacity
     * @param num_kv_heads Number of key-value heads
     * @param head_dim Head dimension
     * @param seq_len Sequence length of new tokens
     * @param dtype Data type
     * @param device Device
     * @param max_position_embeddings Maximum position embeddings (for initial capacity)
     */
    void ensure_capacity(size_t batch_size, size_t num_kv_heads, size_t head_dim, size_t seq_len,
                         infinicore::DataType dtype, const infinicore::Device &device,
                        size_t max_position_embeddings = 4096) {
        size_t required_capacity = seq_len + std::accumulate(cache_positions.begin(), cache_positions.end(), 0, [](int a, int b) { return std::max(a, b); });

        // VALIDATION: Verify input parameters
        if (num_kv_heads == 0 || head_dim == 0 || seq_len == 0) {
            SPDLOG_ERROR("KVCacheLayer::ensure_capacity: Invalid parameters - num_kv_heads: {}, head_dim: {}, seq_len: {}",
                        num_kv_heads, head_dim, seq_len);
            throw std::runtime_error("KV cache ensure_capacity: invalid parameters");
        }

        // Lazy initialization
        if (!initialized) {
            max_capacity = std::max(required_capacity, size_t(4096)); // Start with at least 4096
            k_cache = infinicore::Tensor::empty({batch_size, num_kv_heads, max_capacity, head_dim},
                                                dtype, device);
            v_cache = infinicore::Tensor::empty({batch_size, num_kv_heads, max_capacity, head_dim},
                                                dtype, device);
            cache_positions = std::vector<size_t>(batch_size, 0);
            initialized = true;

            // VALIDATION: Verify cache was created correctly
            // Shape is [batch_size, num_kv_heads, max_capacity, head_dim]
            if (k_cache->shape()[0] != batch_size || k_cache->shape()[1] != num_kv_heads ||
                k_cache->shape()[2] != max_capacity || k_cache->shape()[3] != head_dim) {
                SPDLOG_ERROR("KVCacheLayer::ensure_capacity: Cache shape mismatch after initialization - expected: [{}, {}, {}, {}], got: {}",
                            batch_size, num_kv_heads, max_capacity, head_dim, shape_to_string(k_cache->shape()));
                throw std::runtime_error("KV cache initialization: shape mismatch");
            }
        }
        // Grow cache if needed (similar to DynamicLayer in Python)
        else if (required_capacity > max_capacity) {
            size_t new_capacity = std::max(max_capacity * 2, required_capacity + max_capacity);
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
            // Shape is [batch_size, num_kv_heads, max_capacity, head_dim]
            if (k_cache->shape()[2] != new_capacity) {
                SPDLOG_ERROR("KVCacheLayer::ensure_capacity: New cache capacity mismatch - expected: {}, got: {}",
                            new_capacity, k_cache->shape()[2]);
                throw std::runtime_error("KV cache growth: capacity mismatch");
            }
        }

        // VALIDATION: Final check that capacity is sufficient
        if (required_capacity > max_capacity) {
            SPDLOG_ERROR("KVCacheLayer::ensure_capacity: Capacity still insufficient after growth - required: {}, max_capacity: {}",
                        required_capacity, max_capacity);
            throw std::runtime_error("KV cache ensure_capacity: capacity insufficient");
        }
    }

    KVCacheLayer(size_t max_batch_size, size_t n_kv_head, size_t head_dim, infinicore::DataType dtype, size_t max_seqlen = 4096, infinicore::Device device = infinicore::context::getDevice())
        : max_capacity(max_seqlen), initialized(false) {
        cache_positions = std::vector<size_t>(max_batch_size, 0);
        ensure_capacity(max_batch_size, n_kv_head, head_dim, max_capacity, dtype, device);
    }

    /**
     * @brief Update cache with new key and value states
     * @param k_new New key states [batch_size, n_kv_head, seq_len, head_dim]
     * @param v_new New value states [batch_size, n_kv_head, seq_len, head_dim]
     * @return Tuple of (k_total, v_total) with shape [batch_size, n_kv_head, total_seq_len, head_dim]
     *
     * Note: This method writes to the cache. If using with attention op, the attention op
     * also writes to the cache, so this should be called AFTER attention, not before.
     */
    std::pair<infinicore::Tensor, infinicore::Tensor> update(
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new) {
        if (k_new->ndim() != 4 || v_new->ndim() != 4) {
            throw std::runtime_error("KVCache update: k_new and v_new must be 4D tensors in [batch_size, n_kv_head, seq_len, head_dim] form.");
        }
        size_t batch_size = k_new->shape()[0];
        size_t num_kv_heads = k_new->shape()[1];
        size_t seq_len = k_new->shape()[2];
        size_t head_dim = k_new->shape()[3];

        // Ensure capacity
        ensure_capacity(batch_size, num_kv_heads, head_dim, seq_len,
                        k_new->dtype(), k_new->device());

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
};

/**
 * @brief Model-level KV cache container (similar to DynamicCache in Python)
 *
 * Stores a list of KVCacheLayer objects, one per model layer.
 * This aligns with Python backend's DynamicCache architecture.
 */
class DynamicCache {
public:
    /**
     * @brief Construct DynamicCache with specified number of layers
     *
     * @param num_layers Number of model layers (creates one cache layer per model layer)
     * @param max_position_embeddings Maximum position embeddings (used for initial capacity)
     */
    DynamicCache(size_t num_layers, size_t max_position_embeddings = 4096)
        : layers_(num_layers), max_position_embeddings_(max_position_embeddings) {}

    /**
     * @brief Update cache with new key and value states for a specific layer
     *
     * @param layer_idx Layer index (0-based)
     * @param k_new New key states [batch_size, n_kv_head, seq_len, head_dim]
     * @param v_new New value states [batch_size, n_kv_head, seq_len, head_dim]
     * @return Tuple of (k_total, v_total) with shape [batch_size, n_kv_head, total_seq_len, head_dim]
     *
     * This method updates the cache for the specified layer and returns the
     * accumulated cache up to the current position.
     */
    std::pair<infinicore::Tensor, infinicore::Tensor> update(
        size_t layer_idx,
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new) {
        if (layer_idx >= layers_.size()) {
            SPDLOG_ERROR("DynamicCache::update: layer_idx {} out of range (num_layers: {})",
                        layer_idx, layers_.size());
            throw std::runtime_error("DynamicCache: layer_idx out of range");
        }

        // Update the cache for this layer
        return layers_[layer_idx].update(k_new, v_new);
    }

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
        const infinicore::Tensor &v_new) {
        return update(0, k_new, v_new);
    }

    /**
     * @brief Get the number of layers in this cache
     */
    size_t num_layers() const { return layers_.size(); }

    /**
     * @brief Get cache position for a specific layer
     */
    size_t cache_position(size_t layer_idx) const {
        if (layer_idx >= layers_.size()) {
            throw std::runtime_error("DynamicCache: layer_idx out of range");
        }
        if (layers_[layer_idx].cache_positions.empty()) {
            return 0;
        }
        return layers_[layer_idx].cache_positions[0]; // All batch items should have same position
    }

    /**
     * @brief Get max position embeddings (used for initial capacity)
     */
    size_t max_position_embeddings() const { return max_position_embeddings_; }

    /**
     * @brief Reset cache for all layers (clear cache positions)
     * This should be called when starting a new generation sequence
     */
    void reset() {
        for (auto& layer : layers_) {
            std::fill(layer.cache_positions.begin(), layer.cache_positions.end(), 0);
            // Note: We don't reset initialized flag or clear the cache tensors
            // to avoid reallocation. The cache will be overwritten on next update.
        }
        SPDLOG_INFO("DynamicCache::reset: All layers reset to position 0");
    }

    /**
     * @brief Full reset: clear cache positions
     * This ensures no stale data persists between different generation sequences
     * The cache tensors will be overwritten on next update, so we only need to reset positions
     * Use this when you need to guarantee a completely clean cache state
     */
    void full_reset() {
        for (auto& layer : layers_) {
            std::fill(layer.cache_positions.begin(), layer.cache_positions.end(), 0);
            // Note: We don't zero out tensors as they will be overwritten on next update
            // Resetting positions ensures the cache starts fresh
        }
    }

    /**
     * @brief Access a specific layer's cache (for advanced usage)
     */
    KVCacheLayer& layer(size_t layer_idx) {
        if (layer_idx >= layers_.size()) {
            throw std::runtime_error("DynamicCache: layer_idx out of range");
        }
        return layers_[layer_idx];
    }

    const KVCacheLayer& layer(size_t layer_idx) const {
        if (layer_idx >= layers_.size()) {
            throw std::runtime_error("DynamicCache: layer_idx out of range");
        }
        return layers_[layer_idx];
    }

private:
    std::vector<KVCacheLayer> layers_;
    size_t max_position_embeddings_;
};

} // namespace infinilm::cache
