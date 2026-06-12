#include "kv_cache.hpp"

#include "../global_state/global_state.hpp"
#include "../utils.hpp"

namespace infinilm::cache {
// ==========================
// StaticKVCacheConfig
// ==========================

StaticKVCacheConfig::StaticKVCacheConfig(
    infinicore::Size _max_batch_size,
    infinicore::Size _max_cache_len)
    : max_batch_size_(_max_batch_size),
      max_cache_len_(_max_cache_len) {
}

std::unique_ptr<CacheConfig>
StaticKVCacheConfig::unique_copy() const {
    return std::make_unique<StaticKVCacheConfig>(*this);
}

infinicore::Size
StaticKVCacheConfig::max_batch_size() const {
    return max_batch_size_;
}

infinicore::Size
StaticKVCacheConfig::max_cache_len() const {
    return max_cache_len_;
}

namespace StaticKVCache {

// ==========================
// StaticKVCache
// ==========================
infinicore::Tensor create_layer_kv_cache(
    const infinicore::Size k_dim,
    const infinicore::Size v_dim,
    const infinicore::Size num_k_heads,
    const infinicore::Size num_v_heads,
    const infinicore::Size max_positional_embedding,
    const infinicore::DataType dtype,
    const StaticKVCacheConfig &config) {
    ASSERT((num_k_heads == num_v_heads) && (k_dim == v_dim));

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();

    size_t rank_batch_size = (config.max_batch_size());
    size_t kv_dim = k_dim;

    bool is_kv_replica = (num_k_heads < rank_info.tp_size && num_v_heads < rank_info.tp_size && num_k_heads == num_v_heads && rank_info.tp_size % num_k_heads == 0);

    size_t num_rank_k_heads = is_kv_replica ? 1 : (num_k_heads / rank_info.tp_size);
    size_t num_rank_v_heads = is_kv_replica ? 1 : (num_v_heads / rank_info.tp_size);

    size_t cache_len = (config.max_cache_len() == std::numeric_limits<infinicore::Size>::max() || config.max_cache_len() == 0 ? max_positional_embedding : config.max_cache_len());

    // Allocate KV cache
    infinicore::Tensor kv_cache = infinicore::Tensor::empty(
        {2,
         rank_batch_size,
         num_rank_k_heads,
         cache_len,
         kv_dim},
        dtype,
        rank_info.device);
    set_zeros(kv_cache);

    infinicore::context::syncStream();

    return kv_cache;
}
}; // namespace StaticKVCache

// ==========================
// PagedKVCacheConfig
// ==========================
PagedKVCacheConfig::PagedKVCacheConfig(
    size_t num_blocks,
    size_t block_size)
    : num_blocks_(num_blocks),
      block_size_(block_size) {
}

std::unique_ptr<CacheConfig>
PagedKVCacheConfig::unique_copy() const {
    return std::make_unique<PagedKVCacheConfig>(*this);
}

size_t
PagedKVCacheConfig::num_blocks() const {
    return num_blocks_;
}

size_t
PagedKVCacheConfig::block_size() const {
    return block_size_;
}

namespace PagedKVCache {
// ==========================
// PagedKVCache
// ==========================
infinicore::Tensor create_layer_kv_cache(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::DataType dtype,
    const PagedKVCacheConfig &config) {
    ASSERT((num_k_heads == num_v_heads) && (k_dim == v_dim));

    const engine::distributed::RankInfo &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();

    size_t kv_dim = k_dim;
    bool is_kv_replica = (num_k_heads < rank_info.tp_size && num_v_heads < rank_info.tp_size && num_k_heads == num_v_heads && rank_info.tp_size % num_k_heads == 0);

    size_t num_rank_k_heads = is_kv_replica ? 1 : (num_k_heads / rank_info.tp_size);
    size_t num_rank_v_heads = is_kv_replica ? 1 : (num_v_heads / rank_info.tp_size);

    size_t num_blocks_per_layer = config.num_blocks();
    size_t block_size = config.block_size();

    infinicore::Shape kv_shape;
    if (global_state::get_infinilm_config().attention_backend == backends::AttentionBackend::FLASH_ATTN) {
        // FLASH_ATTN kernel expects BSHD layout
        kv_shape = {2, num_blocks_per_layer, block_size, num_rank_k_heads, k_dim};
    } else {
        kv_shape = {2, num_blocks_per_layer, num_rank_k_heads, block_size, k_dim};
    }

    // [1+1, num_blocks, num_rank_k_heads, block_size, k_dim]
    infinicore::Tensor kv_cache = infinicore::Tensor::empty(
        kv_shape,
        dtype,
        rank_info.device);
    set_zeros(kv_cache);

    infinicore::context::syncStream();

    return kv_cache;
}
}; // namespace PagedKVCache

} // namespace infinilm::cache
