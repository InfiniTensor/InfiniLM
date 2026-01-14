#pragma once

#include "base_cache.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/device.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>

#include <spdlog/spdlog.h>

namespace infinilm::cache {
class StaticKVCacheConfig final : public CacheConfig {
public:
    StaticKVCacheConfig(
        infinicore::Size _max_batch_size = 1,
        infinicore::Size _max_cache_len = std::numeric_limits<infinicore::Size>::max());

    std::unique_ptr<CacheConfig> unique_copy() const override;
    infinicore::Size max_batch_size() const;
    infinicore::Size max_cache_len() const;

private:
    infinicore::Size max_batch_size_;
    infinicore::Size max_cache_len_;
};

class StaticKVCache final : public Cache {
public:
    StaticKVCache(

        infinicore::Size k_dim,
        infinicore::Size v_dim,
        infinicore::Size num_k_heads,
        infinicore::Size num_v_heads,
        infinicore::Size num_layers,
        infinicore::Size max_positional_embedding,
        infinicore::DataType dtype,
        const StaticKVCacheConfig &config,
        const engine::distributed::RankInfo &rank_info);

    /**
     * @brief Update KV cache at a given layer and cache position.
     *
     * @param layer_idx Which transformer layer
     * @param k         [batch, num_rank_k_heads, seq_len, k_dim]
     * @param v         [batch, num_rank_v_heads, seq_len, v_dim]
     * @param cache_pos Sequence position to write
     *
     * @return (full_k, full_v)
     *         full_k: [batch, num_rank_k_heads, cache_pos + seq_len, k_dim]
     *         full_v: [batch, num_rank_v_heads, cache_pos + seq_len, v_dim]
     */
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    update(size_t layer_idx,
           const infinicore::Tensor &k,
           const infinicore::Tensor &v,
           const infinicore::Tensor &past_sequence_lengths);

    ~StaticKVCache() override = default;

private:
    infinicore::Size k_dim_;
    infinicore::Size v_dim_;
    infinicore::Size num_rank_k_heads_;
    infinicore::Size num_rank_v_heads_;
    infinicore::Size rank_batch_size_;
    infinicore::Size cache_len_;
    infinicore::Size rank_num_layers_;
    infinicore::DataType dtype_;

    // [num_layers, max_batch, num_rank_k_heads, max_cache_len, k_dim]
    infinicore::Tensor k_caches_;

    // [num_layers, max_batch, num_rank_v_heads, max_cache_len, v_dim]
    infinicore::Tensor v_caches_;
};

class PagedKVCacheConfig final : public CacheConfig {
public:
    PagedKVCacheConfig(
        size_t num_blocks,
        size_t block_size = 16);

    std::unique_ptr<CacheConfig> unique_copy() const override;
    size_t num_blocks() const;
    size_t block_size() const;

private:
    size_t num_blocks_;
    size_t block_size_;
};

class PagedKVCache final : public Cache {
public:
    PagedKVCache(

        infinicore::Size k_dim,
        infinicore::Size v_dim,
        infinicore::Size num_k_heads,
        infinicore::Size num_v_heads,
        infinicore::Size num_layers,
        infinicore::DataType dtype,
        const PagedKVCacheConfig &config,
        const engine::distributed::RankInfo &rank_info);

    /**
     * @brief Update Paged KV cache at a given layer given slot info for each token.
     *
     * @param layer_idx Which paged attention layer
     * @param k         [num_rank_k_heads, seq_len, k_dim]
     * @param v         [num_rank_v_heads, seq_len, v_dim]
     * @param slot_mapping [seq_len]
     *
     * @return (full_k, full_v)
     *         full_k: [num_blocks, num_rank_k_heads, block_size, k_dim]
     *         full_v: [num_blocks, num_rank_v_heads, block_size, v_dim]
     */
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    update(size_t layer_idx,
           const infinicore::Tensor &k,
           const infinicore::Tensor &v,
           const infinicore::Tensor &slot_mapping);

    /**
     * @brief Get Paged KV cache at a given layer.
     *
     * @param layer_idx Which paged attention layer
     *
     * @return (full_k, full_v)
     *         full_k: [num_blocks, num_rank_k_heads, block_size, k_dim]
     *         full_v: [num_blocks, num_rank_v_heads, block_size, v_dim]
     */
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    get_paged_kv(size_t layer_idx);

    /**
     * @brief Get contiguous KV cache at a given layer, given the request info
     * among a continuous request batch.
     *
     * @param layer_idx Which paged attention layer
     * @param block_tables [num_requests, max_blocks_per_request]
     * @param cache_lens [num_requests]
     * @param input_offsets [num_requests + 1]
     * @param request_id Which request among a continuous batch of requests
     *
     * @return (full_k, full_v)
     *         full_k: [num_rank_k_heads, total_len, k_dim]
     *         full_v: [num_rank_v_heads, total_len, v_dim]
     */
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    get_contiguous_kv(size_t layer_idx,
                      const infinicore::Tensor block_tables,
                      const infinicore::Tensor cache_lens,
                      const infinicore::Tensor input_offsets,
                      size_t request_id = 0);

    ~PagedKVCache() override
        = default;

private:
    infinicore::Size k_dim_;
    infinicore::Size v_dim_;
    infinicore::Size num_rank_k_heads_;
    infinicore::Size num_rank_v_heads_;
    infinicore::Size rank_num_layers_;
    infinicore::DataType dtype_;
    infinicore::Size block_size_;
    infinicore::Size num_blocks_per_layer_;
    // [num_layers, num_blocks, num_rank_k_heads, block_size, k_dim]
    infinicore::Tensor k_caches_;

    // [num_layers, num_blocks, num_rank_v_heads, block_size, v_dim]
    infinicore::Tensor v_caches_;
};

} // namespace infinilm::cache
