#pragma once

#include "base_cache.hpp"
#include <infinicore/dtype.hpp>

#include <limits>
#include <memory>
#include <utility>

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

namespace StaticKVCache {

infinicore::Tensor create_layer_kv_cache(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::Size max_positional_embedding,
    infinicore::DataType dtype,
    const StaticKVCacheConfig &config);

} // namespace StaticKVCache

class PagedKVCacheConfig final : public CacheConfig {
public:
    PagedKVCacheConfig(
        size_t num_blocks,
        size_t block_size = 256,
        size_t max_batch_size = 1);

    std::unique_ptr<CacheConfig> unique_copy() const override;
    size_t num_blocks() const;
    size_t block_size() const;
    size_t max_batch_size() const;

private:
    size_t num_blocks_;
    size_t block_size_;
    size_t max_batch_size_;
};

namespace PagedKVCache {
infinicore::Tensor create_layer_kv_cache(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::DataType dtype,
    const PagedKVCacheConfig &config);

// FP8(E4M3) KV cache scales for one layer: returns `{k_scale, v_scale}`, each F32 with
// shape `[num_blocks, num_rank_kv_heads, block_size]` (one scale per token per kv head;
// the head_dim dimension is reduced by the paged_caching kernel on write). The scale
// layout is logical (block, kv_head, token-in-block) and does not follow the cache
// tensor's element order, so it is identical for the paged (BHSD) and FLASH_ATTN (BSHD)
// cache layouts.
std::pair<infinicore::Tensor, infinicore::Tensor> create_layer_kv_scales(
    infinicore::Size num_kv_heads,
    const PagedKVCacheConfig &config);

} // namespace PagedKVCache

} // namespace infinilm::cache
