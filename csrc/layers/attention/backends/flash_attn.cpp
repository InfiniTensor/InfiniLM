#include "flash_attn.hpp"

#include "../../../utils.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::layers::attention::backends {

FlashAttentionImpl::FlashAttentionImpl(size_t num_heads,
                                       size_t head_size,
                                       float scale,
                                       size_t num_kv_heads,
                                       size_t layer_idx)
    : num_heads_(num_heads),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_size) {
    (void)layer_idx;
}

infinicore::Tensor FlashAttentionImpl::forward(
    const AttentionLayer &layer,
    const infinicore::Tensor &query,
    const infinicore::Tensor &key,
    const infinicore::Tensor &value,
    infinicore::Tensor &kv_cache,
    const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    (void)layer;

    ASSERT(attn_metadata.total_sequence_lengths.has_value());
    ASSERT(attn_metadata.block_tables.has_value());
    ASSERT(attn_metadata.slot_mapping.has_value());

    return infinicore::op::paged_flash_attention(
        query,
        key,
        value,
        kv_cache,
        attn_metadata.total_sequence_lengths.value(),
        attn_metadata.input_offsets,
        attn_metadata.cu_seqlens,
        attn_metadata.block_tables.value(),
        attn_metadata.slot_mapping.value(),
        num_heads_,
        num_kv_heads_,
        head_dim_,
        scale_);
}

} // namespace infinilm::layers::attention::backends
