#include "paged_attn.hpp"

#include "../../../utils.hpp"
#include "../../../utils/agent_debug.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/ops.hpp"
#include "../../../global_state/global_state.hpp"

namespace infinilm::layers::attention::backends {

PagedAttentionImpl::PagedAttentionImpl(size_t num_heads,
                                       size_t head_size,
                                       float scale,
                                       size_t num_kv_heads,
                                       size_t layer_idx)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size) {}

infinicore::Tensor PagedAttentionImpl::forward(const AttentionLayer &layer,
                                               const infinicore::Tensor &query,
                                               const infinicore::Tensor &key,
                                               const infinicore::Tensor &value,
                                               infinicore::Tensor &kv_cache,
                                               const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());

    // #region agent log
    if (layer_idx_ == 0) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "paged_attn.cpp:forward",
            "paged_caching_begin",
            "S",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) +
                ",\"seq_len\":" + std::to_string(query->shape()[0]) +
                ",\"slot_len\":" + std::to_string(slot_mapping.value()->shape()[0]) +
                ",\"num_kv_heads\":" + std::to_string(num_kv_heads_) + "}",
            "post-fix");
    }
    // #endregion

    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value());
    infinicore::context::syncStream();
    // #region agent log
    if (layer_idx_ == 0) {
        const int tp_rank = infinilm::global_state::get_tensor_model_parallel_rank();
        infinilm::agent_debug::log(
            "paged_attn.cpp:forward",
            "paged_caching_done",
            "S",
            std::string("{\"tp_rank\":") + std::to_string(tp_rank) + "}",
            "post-fix");
    }
    // #endregion

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 2. Compute attention
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_heads_, head_dim_}, query->dtype(), query->device());
    if (is_prefill) {
        infinicore::op::paged_attention_prefill_(
            attn_output,
            query,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            input_offsets.value(),
            std::nullopt,
            scale_);
    } else {
        infinicore::op::paged_attention_(
            attn_output,
            query,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            scale_);
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * head_dim_});
    return attn_output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> PagedAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinicore::Tensor key,
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
