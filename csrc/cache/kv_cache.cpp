#include "kv_cache.hpp"

#include "../utils.hpp"

#include <stdexcept>

namespace infinilm::cache {

// ==========================
// StaticKVCache
// ==========================

StaticKVCache::StaticKVCache(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::Size num_layers,
    infinicore::Size max_positional_embedding,
    infinicore::DataType dtype,
    const StaticKVCacheConfig &config,
    const engine::distributed::RankInfo &rank_info)
    : Cache(),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_rank_k_heads_(num_k_heads / rank_info.tp_size),
      num_rank_v_heads_(num_v_heads / rank_info.tp_size),
      rank_batch_size_(config.max_batch_size()),
      cache_len_(std::min(config.max_cache_len(), max_positional_embedding)),
      rank_num_layers_(num_layers),
      dtype_(dtype) {

    // Allocate K cache
    k_caches_ = infinicore::Tensor::empty(
        {rank_num_layers_,
         rank_batch_size_,
         num_rank_k_heads_,
         cache_len_,
         k_dim_},
        dtype_,
        rank_info.device);

    // Allocate V cache
    v_caches_ = infinicore::Tensor::empty(
        {rank_num_layers_,
         rank_batch_size_,
         num_rank_v_heads_,
         cache_len_,
         v_dim_},
        dtype_,
        rank_info.device);

    spdlog::info("Created Static KV Cache: K[{}] V[{}]", k_caches_->info(), v_caches_->info());
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
StaticKVCache::update(size_t layer_idx,
                      const infinicore::Tensor &k,
                      const infinicore::Tensor &v,
                      const infinicore::Tensor &cache_positions) {
    ASSERT(layer_idx < rank_num_layers_);

    auto batch_size = k->size(0);
    auto update_len = k->size(2);
    size_t cache_pos = reinterpret_cast<int64_t *>(cache_positions->to(infinicore::Device::cpu())->data())[0];
    auto result_len = cache_pos + update_len;

    ASSERT(result_len <= cache_len_);

    ASSERT_EQ(batch_size, rank_batch_size_);

    auto k_cache_layer = k_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);
    auto v_cache_layer = v_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);

    auto k_cache_update = k_cache_layer->narrow({{2, cache_pos, update_len}});
    auto v_cache_update = v_cache_layer->narrow({{2, cache_pos, update_len}});

    k_cache_update->copy_from(k);
    v_cache_update->copy_from(v);

    auto k_total = k_cache_layer->narrow({{2, 0, result_len}});
    auto v_total = v_cache_layer->narrow({{2, 0, result_len}});

    return {k_total, v_total};
}

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

} // namespace infinilm::cache
