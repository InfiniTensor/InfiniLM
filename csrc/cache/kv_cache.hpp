#pragma once

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace infinilm::cache {

/**
 * @brief Simple KV cache structure for incremental decoding
 *
 * Stores key and value caches with shape [batch_size, n_kv_head, capacity, head_dim]
 * Similar to DynamicLayer in Python cache_utils.py
 *
 * This is a common component that can be used by any model architecture
 * that needs KV caching for attention mechanisms.
 */
struct KVCache {
    infinicore::Tensor k_cache;          // [batch_size, n_kv_head, capacity, head_dim]
    infinicore::Tensor v_cache;          // [batch_size, n_kv_head, capacity, head_dim]
    std::vector<size_t> cache_positions; // Current position in cache
    size_t max_capacity;                 // Maximum capacity of cache
    bool initialized;                    // Whether cache has been initialized

    KVCache() : max_capacity(0), initialized(false) {}

    /**
     * @brief Initialize or update cache capacity
     * @param num_kv_heads Number of key-value heads
     * @param head_dim Head dimension
     * @param seq_len Sequence length of new tokens
     * @param dtype Data type
     * @param device Device
     */
    void ensure_capacity(size_t batch_size, size_t num_kv_heads, size_t head_dim, size_t seq_len,
                         infinicore::DataType dtype, const infinicore::Device &device) {
        size_t required_capacity = seq_len + std::accumulate(cache_positions.begin(), cache_positions.end(), 0, [](int a, int b) { return std::max(a, b); });

        // Lazy initialization
        if (!initialized) {
            max_capacity = std::max(required_capacity, size_t(4096)); // Start with at least 4096
            k_cache = infinicore::Tensor::empty({batch_size, num_kv_heads, max_capacity, head_dim},
                                                dtype, device);
            v_cache = infinicore::Tensor::empty({batch_size, num_kv_heads, max_capacity, head_dim},
                                                dtype, device);
            cache_positions = std::vector<size_t>(batch_size, 0);
            initialized = true;
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
        }
    }

    KVCache(size_t max_batch_size, size_t n_kv_head, size_t head_dim, infinicore::DataType dtype, size_t max_seqlen = 4096, infinicore::Device device = infinicore::context::getDevice())
        : max_capacity(max_seqlen), initialized(false) {
        cache_positions = std::vector<size_t>(max_batch_size, 0);
        ensure_capacity(max_batch_size, n_kv_head, head_dim, max_capacity, dtype, device);
    }

    /**
     * @brief Update cache with new key and value states
     * @param k_new New key states [batch_size, n_kv_head, seq_len, head_dim]
     * @param v_new New value states [batch_size, n_kv_head, seq_len, head_dim]
     * @return Tuple of (k_total, v_total) with shape [n_kv_head, total_seq_len, head_dim]
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

} // namespace infinilm::cache
