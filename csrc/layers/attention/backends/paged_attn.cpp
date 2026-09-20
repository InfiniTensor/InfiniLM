#include "paged_attn.hpp"

#include "../../../utils.hpp"
#include "infinicore/ops.hpp"

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

    // FP8(E4M3) KV cache: fetch this layer's per-token-per-kv-head scales (allocated with
    // the cache in ForwardContext::kv_scale_vec). Non-FP8 caches pass nullopt, keeping the
    // operator behavior bitwise unchanged.
    std::optional<infinicore::Tensor> k_scale = std::nullopt;
    std::optional<infinicore::Tensor> v_scale = std::nullopt;
    if (kv_cache->dtype() == infinicore::DataType::F8) {
        const auto &kv_scale_vec = infinilm::global_state::get_forward_context().kv_scale_vec;
        if (layer_idx_ >= kv_scale_vec.size()
            || kv_scale_vec[layer_idx_].first.empty()
            || kv_scale_vec[layer_idx_].second.empty()) {
            throw std::runtime_error(
                "infinilm::layers::attention::backends::PagedAttentionImpl: FP8 KV cache requires per-layer "
                "k_scale/v_scale, but none were allocated for layer "
                + std::to_string(layer_idx_)
                + ". FP8 KV cache is currently supported only by the default paged KV cache allocation path.");
        }
        k_scale = kv_scale_vec[layer_idx_].first;
        v_scale = kv_scale_vec[layer_idx_].second;
    }

    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value(), k_scale, v_scale);

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 2. Compute attention
    const size_t value_head_dim = value->size(value->ndim() - 1);
    infinicore::Tensor attn_output = infinicore::Tensor::empty({seq_len, num_heads_, value_head_dim}, query->dtype(), query->device());
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
            scale_,
            k_scale,
            v_scale);
    } else {
        infinicore::op::paged_attention_(
            attn_output,
            query,
            k_total,
            v_total,
            block_tables.value(),
            total_sequence_lengths.value(),
            std::nullopt,
            scale_,
            k_scale,
            v_scale);
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * value_head_dim});
    return attn_output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> PagedAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinicore::Tensor key,
                                                                                          const infinicore::Tensor value,
                                                                                          infinicore::Tensor &kv_cache,
                                                                                          const infinicore::Tensor slot_mapping,
                                                                                          const std::optional<infinicore::Tensor> &k_scale,
                                                                                          const std::optional<infinicore::Tensor> &v_scale) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    v_cache_layer = v_cache_layer->narrow({{3, 0, value->size(value->ndim() - 1)}});
    infinicore::op::paged_caching_(
        k_cache_layer,
        v_cache_layer,
        key,
        value,
        slot_mapping,
        k_scale,
        v_scale);

    return {k_cache_layer, v_cache_layer};
}
} // namespace infinilm::layers::attention::backends
