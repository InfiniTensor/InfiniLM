#include "flash_attn.hpp"

#include "../../../global_state/global_state.hpp"
#include "../../../utils.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include <string>

namespace infinilm::layers::attention::backends {

FlashAttentionImpl::FlashAttentionImpl(size_t num_heads,
                                       size_t head_size,
                                       float scale,
                                       size_t num_kv_heads,
                                       size_t layer_idx,
                                       const infinicore::Device &device)
    : num_heads_(num_heads),
      head_size_(head_size),
      scale_(scale),
      num_kv_heads_(num_kv_heads),
      layer_idx_(layer_idx),
      head_dim_(head_size),
      device_(device) {

    const infinilm::global_state::InfinilmConfig &infinilm_config = infinilm::global_state::get_infinilm_config();
    if (!infinilm_config.model_config) {
        throw std::runtime_error("infinilm::layers::attention::backends::FlashAttentionImpl: model_config is null");
    }

    const auto &model_config = infinilm_config.model_config;
    dtype_ = model_config->get_dtype();
    max_position_embeddings_ = model_config->get<size_t>("max_position_embeddings");

    enable_workspace_manager_ = infinilm_config.enable_workspace_manager;
    if (enable_workspace_manager_) {
        this->_register_inference_buffer();
    }
}

infinicore::Tensor FlashAttentionImpl::forward(const AttentionLayer &layer,
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
    infinicore::Tensor attn_output;
    if (is_prefill) {
        if (enable_workspace_manager_) {
            attn_output = max_attn_output_->narrow({{0, 0, seq_len}});
        } else {
            attn_output = infinicore::Tensor::empty({seq_len, num_heads_, head_dim_}, dtype_, device_);
        }
        infinicore::op::mha_varlen_(
            attn_output,
            query,
            k_total,
            v_total,
            input_offsets.value(),
            cu_seqlens.value(),
            block_tables.value(),
            max_position_embeddings_,
            max_position_embeddings_,
            std::nullopt,
            scale_);
    } else {
        // FA2 decode path: flash::mha_fwd_kvcache
        // In paged-attn mode, seq_len = actual batch_size (one query token per sequence).
        // q_reshaped: [seq_len, num_heads, head_dim] → [seq_len, 1, num_heads, head_dim]
        // k/v cache:  [num_blocks, block_size, num_kv_heads, head_dim]
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

std::tuple<infinicore::Tensor, infinicore::Tensor> FlashAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                          const infinicore::Tensor key,
                                                                                          const infinicore::Tensor value,
                                                                                          infinicore::Tensor &kv_cache,
                                                                                          const infinicore::Tensor slot_mapping) const {
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    infinicore::op::paged_caching_(
        k_cache_layer->permute({0, 2, 1, 3}), // permute to BHSD for paged_caching_
        v_cache_layer->permute({0, 2, 1, 3}),
        key,
        value,
        slot_mapping);

    return {k_cache_layer, v_cache_layer};
}

void FlashAttentionImpl::_register_inference_buffer() {
    const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
    auto &workspace_manager = infinilm::global_state::get_forward_context().workspace_manager;
    const size_t max_num_batched_tokens = infinilm_config.max_num_batched_tokens;

    const std::string cache_key = std::string("FlashAttentionImpl_max_num_batched_tokens_")
                                + std::to_string(max_num_batched_tokens) + "_num_heads_"
                                + std::to_string(num_heads_) + "_head_dim_"
                                + std::to_string(head_dim_) + "_dtype_"
                                + infinicore::toString(dtype_) + "_device_"
                                + device_.toString();

    const infinicore::Shape flash_attn_buffer_shape = {max_num_batched_tokens, num_heads_, head_dim_};
    workspace_manager.register_buffer(
        cache_key,
        flash_attn_buffer_shape,
        dtype_,
        device_,
        [this, max_num_batched_tokens](const infinicore::Tensor &flash_attention_impl_buffer) {
            const auto buffer_shape = flash_attention_impl_buffer->shape();
            ASSERT(buffer_shape[0] == max_num_batched_tokens && buffer_shape[1] == num_heads_ && buffer_shape[2] == head_dim_);
            max_attn_output_ = flash_attention_impl_buffer;
        });
}
} // namespace infinilm::layers::attention::backends
