#include "flash_decode_attn.hpp"

#include "../../../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_kvcache.hpp"

namespace infinilm::layers::attention::backends {

FlashDecodeAttentionImpl::FlashDecodeAttentionImpl(size_t num_heads,
                                                      size_t head_size,
                                                      float scale,
                                                      size_t num_kv_heads,
                                                      size_t layer_idx)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size) {

    const infinilm::global_state::InfinilmConfig &infinilm_config = infinilm::global_state::get_infinilm_config();
    if (!infinilm_config.model_config) {
        throw std::runtime_error("infinilm::layers::attention::backends::FlashDecodeAttentionImpl: model_config is null");
    }
    max_position_embeddings_ = infinilm_config.model_config->get<size_t>("max_position_embeddings");
}

infinicore::Tensor FlashDecodeAttentionImpl::forward(const AttentionLayer &layer,
                                                       const infinicore::Tensor &query,
                                                       const infinicore::Tensor &key,
                                                       const infinicore::Tensor &value,
                                                       infinicore::Tensor &kv_cache,
                                                       const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    auto cu_seqlens = attn_metadata.cu_seqlens;

    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());

    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value());

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 2. Compute attention
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_heads_, head_dim_}, query->dtype(), query->device());
    if (is_prefill) {
        infinicore::op::paged_attention_prefill_(
            attn_output,
            query,
            k_total->permute({0, 2, 1, 3}),
            v_total->permute({0, 2, 1, 3}),
            block_tables.value(),
            total_sequence_lengths.value(),
            input_offsets.value(),
            std::nullopt,
            scale_);
    } else {
        auto q_for_fa = query->view({seq_len, 1, num_heads_, head_dim_});
        auto attn_out_4d = infinicore::op::mha_kvcache(
            q_for_fa,
            k_total, // [num_blocks, block_size, num_kv_heads, head_dim]
            v_total,
            total_sequence_lengths.value(), // [seq_len] int32 (one entry per sequence)
            block_tables.value(),           // [seq_len, max_num_blocks_per_seq] int32
            std::nullopt,
            scale_);
        attn_output = attn_out_4d->view({seq_len, num_heads_, head_dim_});
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * head_dim_});
    return attn_output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> FlashDecodeAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinicore::Tensor key,
                                                                                          const infinicore::Tensor value,
                                                                                          infinicore::Tensor &kv_cache,
                                                                                          const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    infinicore::op::paged_caching_(
        k_cache_layer->permute({0, 2, 1, 3}),
        v_cache_layer->permute({0, 2, 1, 3}),
        key,
        value,
        slot_mapping);

    return {k_cache_layer, v_cache_layer};
}
} // namespace infinilm::layers::attention::backends
