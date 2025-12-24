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
           const infinicore::Tensor &cache_positions);

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

} // namespace infinilm::cache
