#include "paged_attn.hpp"

#include "../../../global_state/global_state.hpp"
#include "../../../utils.hpp"
#include "attention_layer.hpp"
#include "infinicore/ops.hpp"

#include <string>

namespace infinilm::layers::attention::backends {

PagedAttentionImpl::PagedAttentionImpl(size_t num_heads,
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
        throw std::runtime_error("infinilm::layers::attention::backends::PagedAttentionImpl: model_config is null");
    }

    dtype_ = infinilm_config.model_config->get_dtype();

    this->_initialize_preallocated_workspace();
}

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

    // 1. update paged kv cache
    auto [k_total, v_total] = do_kv_cache_update(layer, key, value, kv_cache, slot_mapping.value());

    size_t seq_len = query->shape()[0];
    bool is_prefill = (seq_len != total_sequence_lengths.value()->shape()[0]);

    // 2. Compute attention
    infinicore::Tensor attn_output = max_attn_output_->narrow({{0, 0, seq_len}});
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

void PagedAttentionImpl::_initialize_preallocated_workspace() {
    const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
    auto &preallocated_workspace = infinilm::global_state::get_forward_context().preallocated_workspace;
    const size_t max_num_batched_tokens = infinilm_config.max_num_batched_tokens;

    const std::string cache_key = std::string("PagedAttentionImpl_max_num_batched_tokens_")
                                + std::to_string(max_num_batched_tokens) + "_num_heads_"
                                + std::to_string(num_heads_) + "_head_dim_"
                                + std::to_string(head_dim_) + "_dtype_"
                                + infinicore::toString(dtype_) + "_device_"
                                + device_.toString();

    if (preallocated_workspace.find(cache_key) == preallocated_workspace.end()) {
        auto paged_attention_impl_buffer = infinicore::Tensor::empty({max_num_batched_tokens, num_heads_, head_dim_}, dtype_, device_);
        preallocated_workspace[cache_key] = paged_attention_impl_buffer;
    }

    auto paged_attention_impl_buffer = preallocated_workspace.at(cache_key);
    const auto buffer_shape = paged_attention_impl_buffer->shape();
    ASSERT(buffer_shape[0] == max_num_batched_tokens && buffer_shape[1] == num_heads_ && buffer_shape[2] == head_dim_);

    max_attn_output_ = paged_attention_impl_buffer;
}

} // namespace infinilm::layers::attention::backends
