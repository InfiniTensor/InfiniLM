#include "kv_cache.hpp"

#include "../utils.hpp"
#include "infinicore/ops.hpp"
#include <stdexcept>

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
      cache_len_(config.max_cache_len() == std::numeric_limits<infinicore::Size>::max() || config.max_cache_len() == 0 ? max_positional_embedding : config.max_cache_len()),
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
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
StaticKVCache::update(size_t layer_idx,
                      const infinicore::Tensor &k,
                      const infinicore::Tensor &v,
                      const infinicore::Tensor &past_sequence_lengths) {
    ASSERT(layer_idx < rank_num_layers_);

    auto batch_size = k->size(0);
    auto update_len = k->size(2);
    size_t cache_pos = reinterpret_cast<int64_t *>(past_sequence_lengths->to(infinicore::Device::cpu())->data())[0];
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

// ==========================
// PagedKVCache
// ==========================
PagedKVCache::PagedKVCache(
    infinicore::Size k_dim,
    infinicore::Size v_dim,
    infinicore::Size num_k_heads,
    infinicore::Size num_v_heads,
    infinicore::Size num_layers,
    infinicore::DataType dtype,
    const PagedKVCacheConfig &config,
    const engine::distributed::RankInfo &rank_info)
    : Cache(),
      k_dim_(k_dim),
      v_dim_(v_dim),
      num_rank_k_heads_(num_k_heads / rank_info.tp_size),
      num_rank_v_heads_(num_v_heads / rank_info.tp_size),
      rank_num_layers_(num_layers),
      dtype_(dtype),
      num_blocks_per_layer_(config.num_blocks()),
      block_size_(config.block_size()) {
    // [num_layers, num_blocks, num_rank_k_heads, block_size, k_dim]
    k_caches_ = infinicore::Tensor::empty(
        {rank_num_layers_,
         num_blocks_per_layer_,
         num_rank_k_heads_,
         block_size_,
         k_dim_},
        dtype_,
        rank_info.device);

    // [num_layers, num_blocks, num_rank_v_heads, block_size, v_dim]
    v_caches_ = infinicore::Tensor::empty(
        {rank_num_layers_,
         num_blocks_per_layer_,
         num_rank_v_heads_,
         block_size_,
         v_dim_},
        dtype_,
        rank_info.device);
}

std::tuple<infinicore::Tensor, infinicore::Tensor> PagedKVCache::update(
    size_t layer_idx,
    const infinicore::Tensor &k,
    const infinicore::Tensor &v,
    const infinicore::Tensor &slot_mapping) {

    auto &&[k_cache_layer, v_cache_layer] = get_paged_kv(layer_idx);

    infinicore::op::paged_caching_(
        k_cache_layer,
        v_cache_layer,
        k,
        v,
        slot_mapping);
    return {k_cache_layer, v_cache_layer};
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
PagedKVCache::get_paged_kv(size_t layer_idx) {
    auto k_cache_layer = k_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);
    auto v_cache_layer = v_caches_->narrow({{0, layer_idx, 1}})->squeeze(0);
    return {k_cache_layer, v_cache_layer};
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
PagedKVCache::get_contiguous_kv(
    size_t layer_idx,
    const infinicore::Tensor block_tables,
    const infinicore::Tensor cache_lens,
    const infinicore::Tensor input_offsets,
    size_t request_id) {
    ASSERT_EQ(block_tables->dtype(), infinicore::DataType::I64);
    ASSERT_EQ(cache_lens->dtype(), infinicore::DataType::I64);
    ASSERT_EQ(input_offsets->dtype(), infinicore::DataType::I64);

    auto nreq = block_tables->size(0);
    auto block_tables_cpu = block_tables->to(infinicore::Device::cpu());
    auto cache_lens_cpu = cache_lens->to(infinicore::Device::cpu());
    auto input_offsets_cpu = input_offsets->to(infinicore::Device::cpu());
    infinicore::context::syncDevice();

    // [num_blocks, num_rank_v_heads, block_size, v_dim]
    auto &&[k_cache_layer, v_cache_layer] = get_paged_kv(layer_idx);

    auto req = request_id;
    auto cache_lens_ptr = reinterpret_cast<const int64_t *>(cache_lens_cpu->data());
    auto input_offsets_ptr = reinterpret_cast<const int64_t *>(input_offsets_cpu->data());
    int64_t total_len = cache_lens_ptr[req] + (input_offsets_ptr[req + 1] - input_offsets_ptr[req]);

    auto full_k = infinicore::Tensor::empty(
        {num_rank_k_heads_, (size_t)total_len, k_dim_},
        k_cache_layer->dtype(), k_cache_layer->device());

    auto full_v = infinicore::Tensor::empty(
        {num_rank_v_heads_, (size_t)total_len, v_dim_},
        v_cache_layer->dtype(), v_cache_layer->device());

    size_t nblocks = total_len / block_size_;
    size_t r = total_len % block_size_;

    for (size_t b = 0; b < nblocks; b++) {
        size_t bid = *((int64_t *)(block_tables_cpu->narrow({{0, req, 1}, {1, b, 1}})->data()));

        full_k->narrow({{1, b * block_size_, block_size_}})
            ->copy_from(k_cache_layer->narrow({{0, bid, 1}})->squeeze(0));
        full_v->narrow({{1, b * block_size_, block_size_}})
            ->copy_from(v_cache_layer->narrow({{0, bid, 1}})->squeeze(0));
    }

    if (r > 0) {
        size_t bid = *((int64_t *)(block_tables_cpu->narrow({{0, req, 1}, {1, nblocks, 1}})->data()));

        full_k->narrow({{1, nblocks * block_size_, r}})
            ->copy_from(k_cache_layer->narrow({{0, bid, 1}})->squeeze(0)->narrow({{1, 0, r}}));
        full_v->narrow({{1, nblocks * block_size_, r}})
            ->copy_from(v_cache_layer->narrow({{0, bid, 1}})->squeeze(0)->narrow({{1, 0, r}}));
    }

    return {full_k, full_v};
}

} // namespace infinilm::cache
