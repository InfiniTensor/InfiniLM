#include "flash_attn.hpp"
#include "../../../backends/operators/operators.hpp"
#include "../../../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cstdlib>
#include <cstring>

namespace {
bool should_use_atb_decode_attention() {
    const char *impl = std::getenv("INFINILM_ASCEND_DECODE_ATTENTION_IMPL");
    return impl != nullptr && std::strcmp(impl, "atb") == 0;
}
} // namespace

namespace infinilm::layers::attention::backends {
FlashAttentionImpl::FlashAttentionImpl(size_t num_heads, size_t head_size, float scale, size_t num_kv_heads,
                                       size_t layer_idx)
    : num_heads_(num_heads), head_size_(head_size), scale_(scale), num_kv_heads_(num_kv_heads), layer_idx_(layer_idx),
      head_dim_(head_size) {
    const infinilm::global_state::InfinilmConfig &infinilm_config = infinilm::global_state::get_infinilm_config();
    if (!infinilm_config.model_config) {
        throw std::runtime_error("infinilm::layers::attention::backends::FlashAttentionImpl: model_config is null");
    }
    max_position_embeddings_ = infinilm_config.model_config->get<size_t>("max_position_embeddings");
}
infinicore::Tensor FlashAttentionImpl::forward(const AttentionLayer &layer, const infinicore::Tensor &query,
                                               const infinicore::Tensor &key, const infinicore::Tensor &value,
                                               infinicore::Tensor &kv_cache,
                                               const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;
    auto input_offsets = attn_metadata.input_offsets;
    auto block_tables = attn_metadata.block_tables;
    auto slot_mapping = attn_metadata.slot_mapping;
    auto cu_seqlens = attn_metadata.cu_seqlens;
    auto total_sequence_lengths_host = attn_metadata.total_sequence_lengths_host;
    auto block_tables_host = attn_metadata.block_tables_host;
    ASSERT(block_tables.has_value());
    ASSERT(slot_mapping.has_value());
    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value());
    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);
    // 2. Compute attention
    infinicore::Tensor attn_output =
        infinicore::Tensor::empty({seq_len, num_heads_, head_dim_}, query->dtype(), query->device());
    // k_total/v_total come from kv_cache narrow+squeeze (no permute), so layout
    // is `[num_blocks, block_size, num_kv_heads, head_dim]` (BSHD), which is
    // exactly what flash-attn's mha_varlen_fwd / mha_fwd_kvcache expect.
    if (is_prefill) {
#ifdef INFINILM_ENABLE_INFINIOPS
        if (infinilm::backends::ops::should_use(query->device())) {
            infinilm::backends::ops::mha_varlen_fwd(attn_output, query, k_total, v_total, input_offsets.value(),
                                                    cu_seqlens.value(), block_tables.value(), max_position_embeddings_,
                                                    max_position_embeddings_, scale_);
        } else
#endif
            infinilm::backends::ops::mha_varlen_(attn_output, query, k_total, v_total, input_offsets.value(),
                                                 cu_seqlens.value(), block_tables.value(), max_position_embeddings_,
                                                 max_position_embeddings_, std::nullopt, scale_);
    } else {
        // FA2 decode path: flash::mha_fwd_kvcache
        // In paged-attn mode, seq_len = actual batch_size (one query token per sequence).
        // q_reshaped: [seq_len, num_heads, head_dim] → [seq_len, 1, num_heads, head_dim]
        auto q_for_fa = query->view({seq_len, 1, num_heads_, head_dim_});
#ifdef INFINILM_ENABLE_INFINIOPS
        if (infinilm::backends::ops::should_use(query->device())) {
            if (should_use_atb_decode_attention()) {
                if (infinicore::context::isGraphRecording() &&
                    (!total_sequence_lengths_host.has_value() || !block_tables_host.has_value())) {
                    throw std::runtime_error("Ascend graph decode requires CPU mirrors for paged attention metadata.");
                }
                infinilm::backends::ops::paged_attention(
                    attn_output, query, k_total, v_total, total_sequence_lengths.value(), block_tables.value(),
                    static_cast<int64_t>(num_heads_), static_cast<int64_t>(num_kv_heads_),
                    static_cast<int64_t>(head_dim_), scale_, total_sequence_lengths_host, block_tables_host);
            } else {
                // Default Ascend decode uses ACLNN FIA V4.  Set
                // `INFINILM_ASCEND_DECODE_ATTENTION_IMPL=atb` only for
                // focused ATB comparison.
                if (infinicore::context::isGraphRecording() && !total_sequence_lengths_host.has_value()) {
                    throw std::runtime_error(
                        "Ascend graph decode with ACLNN FIA requires a CPU mirror for sequence lengths.");
                }
                auto attn_out_4d = infinicore::Tensor::empty(q_for_fa->shape(), q_for_fa->dtype(), q_for_fa->device());
                infinilm::backends::ops::mha_fwd_kvcache(attn_out_4d, q_for_fa, k_total, v_total,
                                                         total_sequence_lengths.value(), block_tables.value(), scale_,
                                                         total_sequence_lengths_host);
                attn_output = attn_out_4d->view({seq_len, num_heads_, head_dim_});
            }
        } else {
#endif
            auto attn_out_4d = infinilm::backends::ops::mha_kvcache(
                q_for_fa,
                k_total, // [num_blocks, block_size, num_kv_heads, head_dim]
                v_total,
                total_sequence_lengths.value(), // [seq_len] int32 (one entry per sequence)
                block_tables.value(),           // [seq_len, max_num_blocks_per_seq] int32
                std::nullopt, scale_);
            attn_output = attn_out_4d->view({seq_len, num_heads_, head_dim_});
#ifdef INFINILM_ENABLE_INFINIOPS
        }
#endif
    }
    attn_output = attn_output->view({1, seq_len, num_heads_ * head_dim_});
    return attn_output;
}
std::tuple<infinicore::Tensor, infinicore::Tensor>
FlashAttentionImpl::do_kv_cache_update(const AttentionLayer &layer, const infinicore::Tensor key,
                                       const infinicore::Tensor value, infinicore::Tensor &kv_cache,
                                       const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
#ifdef INFINILM_ENABLE_INFINIOPS
    if (infinilm::backends::ops::should_use(key->device())) {
        infinilm::backends::ops::reshape_and_cache(key, value, kv_cache, slot_mapping);
        return {k_cache_layer, v_cache_layer};
    }
#endif
    infinilm::backends::ops::paged_caching_(k_cache_layer->permute({0, 2, 1, 3}), // permute to BHSD for paged_caching_
                                            v_cache_layer->permute({0, 2, 1, 3}), key, value, slot_mapping);
    return {k_cache_layer, v_cache_layer};
}
} // namespace infinilm::layers::attention::backends
