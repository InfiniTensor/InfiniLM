#include "hybrid_attn.hpp"

#include "../../../utils.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/paged_attention.hpp"
#include "infinicore/ops/paged_caching.hpp"

#include <stdexcept>

namespace infinilm::layers::attention::backends {

HybridAttentionImpl::HybridAttentionImpl(size_t num_heads,
                                         size_t head_size,
                                         float scale,
                                         size_t num_kv_heads,
                                         size_t layer_idx)
    : num_heads_(num_heads),
      scale_(scale),
      head_dim_(head_size) {

    const infinilm::global_state::InfinilmConfig &infinilm_config = infinilm::global_state::get_infinilm_config();
    if (!infinilm_config.model_config) {
        throw std::runtime_error("infinilm::layers::attention::backends::HybridAttentionImpl: model_config is null");
    }
    max_position_embeddings_ = infinilm_config.model_config->get<size_t>("max_position_embeddings");
}

infinicore::Tensor HybridAttentionImpl::forward(const AttentionLayer &layer,
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

    ASSERT(total_sequence_lengths.has_value());
    ASSERT(input_offsets.has_value());
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());
    ASSERT(cu_seqlens.has_value());

    auto [k_total, v_total] = do_kv_cache_update(key, value, kv_cache, slot_mapping.value());

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_heads_, head_dim_}, query->dtype(), query->device());
    if (is_prefill) {
        infinicore::op::mha_varlen_(
            attn_output,
            query,
            key,
            value,
            input_offsets.value(),
            cu_seqlens.value(),
            block_tables.value(),
            max_position_embeddings_,
            max_position_embeddings_,
            std::nullopt,
            scale_);
    } else {
        infinicore::op::paged_attention_(
            attn_output,
            query,
            k_total->permute({0, 2, 1, 3}),
            v_total->permute({0, 2, 1, 3}),
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            scale_);
    }
    return attn_output->view({1, seq_len, num_heads_ * head_dim_});
}

std::tuple<infinicore::Tensor, infinicore::Tensor> HybridAttentionImpl::do_kv_cache_update(const infinicore::Tensor key,
                                                                                           const infinicore::Tensor value,
                                                                                           infinicore::Tensor &kv_cache,
                                                                                           const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    infinicore::op::paged_caching_(
        k_cache_layer,
        v_cache_layer,
        key,
        value,
        slot_mapping);

    return {k_cache_layer, v_cache_layer};
}

} // namespace infinilm::layers::attention::backends
