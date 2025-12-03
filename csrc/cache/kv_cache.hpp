#pragma once

#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"
#include <algorithm>
#include <utility>
#include <memory>

namespace infinilm::cache {

/**
 * @brief Simple KV cache structure for incremental decoding
 *
 * Stores key and value caches with shape [n_kv_head, capacity, head_dim]
 * Similar to DynamicLayer in Python cache_utils.py
 *
 * This is a common component that can be used by any model architecture
 * that needs KV caching for attention mechanisms.
 */
struct KVCache {
    infinicore::Tensor k_cache;  // [n_kv_head, capacity, head_dim]
    infinicore::Tensor v_cache;  // [n_kv_head, capacity, head_dim]
    size_t cache_position;        // Current position in cache
    size_t max_capacity;          // Maximum capacity of cache
    bool initialized;             // Whether cache has been initialized

    KVCache()
        : cache_position(0), max_capacity(0), initialized(false),
          // Create empty placeholder tensors (will be replaced on first use)
          k_cache(infinicore::Tensor::empty({1, 1, 1}, infinicore::DataType::F32,
                                            infinicore::Device(infinicore::Device::Type::CPU, 0))),
          v_cache(infinicore::Tensor::empty({1, 1, 1}, infinicore::DataType::F32,
                                            infinicore::Device(infinicore::Device::Type::CPU, 0))) {}

    /**
     * @brief Initialize or update cache capacity
     * @param num_kv_heads Number of key-value heads
     * @param head_dim Head dimension
     * @param seq_len Sequence length of new tokens
     * @param dtype Data type
     * @param device Device
     */
    void ensure_capacity(size_t num_kv_heads, size_t head_dim, size_t seq_len,
                        infinicore::DataType dtype, const infinicore::Device &device) {
        size_t required_capacity = cache_position + seq_len;

        // Lazy initialization
        if (!initialized) {
            max_capacity = std::max(required_capacity, size_t(4096));  // Start with at least 4096
            k_cache = infinicore::Tensor::empty({num_kv_heads, max_capacity, head_dim},
                                                dtype, device);
            v_cache = infinicore::Tensor::empty({num_kv_heads, max_capacity, head_dim},
                                                dtype, device);
            cache_position = 0;
            initialized = true;
        }
        // Grow cache if needed (similar to DynamicLayer in Python)
        else if (required_capacity > max_capacity) {
            size_t new_capacity = std::max(max_capacity * 2, required_capacity);
            auto k_new = infinicore::Tensor::empty({num_kv_heads, new_capacity, head_dim},
                                                   dtype, device);
            auto v_new = infinicore::Tensor::empty({num_kv_heads, new_capacity, head_dim},
                                                   dtype, device);

            // Copy existing cache data
            if (cache_position > 0) {
                auto k_slice = k_cache->narrow({{1, 0, cache_position}});
                auto v_slice = v_cache->narrow({{1, 0, cache_position}});
                k_new->narrow({{1, 0, cache_position}})->copy_from(k_slice);
                v_new->narrow({{1, 0, cache_position}})->copy_from(v_slice);
            }

            k_cache = k_new;
            v_cache = v_new;
            max_capacity = new_capacity;
        }
    }

    /**
     * @brief Update cache with new key and value states
     * @param k_new New key states [n_kv_head, seq_len, head_dim]
     * @param v_new New value states [n_kv_head, seq_len, head_dim]
     * @return Tuple of (k_total, v_total) with shape [n_kv_head, total_seq_len, head_dim]
     *
     * Note: This method writes to the cache. If using with attention op, the attention op
     * also writes to the cache, so this should be called AFTER attention, not before.
     */
    std::pair<infinicore::Tensor, infinicore::Tensor> update(
        const infinicore::Tensor &k_new,
        const infinicore::Tensor &v_new) {
        size_t seq_len = k_new->shape()[1];
        size_t num_kv_heads = k_new->shape()[0];
        size_t head_dim = k_new->shape()[2];

        // Ensure capacity
        ensure_capacity(num_kv_heads, head_dim, seq_len,
                       k_new->dtype(), k_new->device());

        // Copy new k/v into cache at current position
        auto k_dst = k_cache->narrow({{1, cache_position, seq_len}});
        auto v_dst = v_cache->narrow({{1, cache_position, seq_len}});
        k_dst->copy_from(k_new);
        v_dst->copy_from(v_new);

        // Update position
        cache_position += seq_len;

        // Return the total cache up to current position
        auto k_total = k_cache->narrow({{1, 0, cache_position}});
        auto v_total = v_cache->narrow({{1, 0, cache_position}});

        return std::make_pair(k_total->contiguous(), v_total->contiguous());
    }
};

} // namespace infinilm::models::common
