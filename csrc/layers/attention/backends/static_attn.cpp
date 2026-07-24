#include "static_attn.hpp"
#include "../../../utils.hpp"
#include "attention_layer.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/per_tensor_dequant_i8.hpp"
#include "infinicore/ops/per_tensor_quant_i8.hpp"

namespace infinilm::layers::attention::backends {

StaticAttentionImpl::StaticAttentionImpl(size_t num_heads,
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
    kv_quant_scheme_ = infinilm::global_state::get_infinilm_config().model_config->get_kv_quant_scheme();
}

infinicore::Tensor StaticAttentionImpl::forward(const AttentionLayer &layer,
                                                infinicore::Tensor &q_rope,
                                                infinicore::Tensor &k_reshaped,
                                                infinicore::Tensor &v_reshaped,
                                                infinicore::Tensor &kv_cache,
                                                const infinilm::global_state::AttentionMetadata &attn_metadata) const {

    auto k_scale = layer.get_k_scale();
    auto v_scale = layer.get_v_scale();
    if (infinilm::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
        infinilm::KVQuantUtils::quantize(
            k_reshaped, v_reshaped,
            this->kv_quant_scheme_,
            k_scale,
            v_scale);
    }

    auto q_reshaped = q_rope->permute({0, 2, 1, 3});     // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]

    //  Prepare Attn
    auto shape = q_reshaped->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[2];

    auto past_sequence_lengths = attn_metadata.past_sequence_lengths;
    auto total_sequence_lengths = attn_metadata.total_sequence_lengths;

    if (infinicore::context::isGraphRecording()) {
        ASSERT(this->kv_quant_scheme_ == infinilm::quantization::KVQuantAlgo::NONE);
        return forward_graph_(q_reshaped, k_permuted, v_permuted, kv_cache, attn_metadata);
    }

    // update static kv cache
    // k_total:  [bs, n_kv_head, max_seq_len, head_dim]
    // v_total : [bs, n_kv_head, max_seq_len, head_dim]
    auto [k_total, v_total] = do_kv_cache_update(layer, k_permuted, v_permuted, kv_cache, past_sequence_lengths.value());

    size_t total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device{infinicore::Device::Type::kCpu})->data())[0];

    if (infinilm::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
        infinilm::KVQuantUtils::dequantize(
            k_total, v_total,
            this->kv_quant_scheme_,
            k_scale,
            v_scale,
            q_reshaped);
    }

    k_total = k_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]
    v_total = v_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]

    // Compute attention.
    size_t ngroup = num_heads_ / num_kv_heads_;
    auto Q = q_reshaped->contiguous()->view({batch_size * num_kv_heads_, ngroup * seq_len, head_dim_});
    auto K = k_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});
    auto V = v_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});

    auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

    auto attn_weight = infinicore::op::matmul(Q, K_transposed, scale_); // [bs * n_kv_head, ng * seq_len, total_seq_len]

    auto attn_weight_softmax = attn_weight->view({batch_size * num_heads_, seq_len, total_seq_len});
    infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

    auto out = infinicore::op::matmul(attn_weight, V); // [bs * n_kv_head, ng * seq_len, head_dim]

    return out->view({batch_size, num_heads_, seq_len, head_dim_})
        ->permute({0, 2, 1, 3})
        ->contiguous()
        ->view({batch_size, seq_len, num_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
}

infinicore::Tensor StaticAttentionImpl::forward_graph_(
    const infinicore::Tensor &query,
    const infinicore::Tensor &key,
    const infinicore::Tensor &value,
    infinicore::Tensor &kv_cache,
    const infinilm::global_state::AttentionMetadata &attn_metadata) const {
    ASSERT_EQ(query->size(2), 1);
    ASSERT(attn_metadata.block_tables.has_value());
    ASSERT(attn_metadata.past_sequence_lengths.has_value());
    ASSERT(attn_metadata.total_sequence_lengths.has_value());

    auto k_cache = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache = kv_cache->narrow({{0, 1, 1}})->squeeze(0);
    infinicore::op::kv_caching_(
        k_cache,
        v_cache,
        key,
        value,
        attn_metadata.past_sequence_lengths.value());

    auto q_decode = query->contiguous()->view({query->size(0), query->size(1), query->size(3)});
    auto output = infinicore::Tensor::empty(q_decode->shape(), q_decode->dtype(), q_decode->device());
    infinicore::op::paged_attention_(
        output,
        q_decode,
        k_cache,
        v_cache,
        attn_metadata.block_tables.value(),
        attn_metadata.total_sequence_lengths.value(),
        std::nullopt,
        scale_);

    return output->view({query->size(0), 1, num_heads_ * head_dim_});
}

std::tuple<infinicore::Tensor, infinicore::Tensor> StaticAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                           const infinicore::Tensor key,
                                                                                           const infinicore::Tensor value,
                                                                                           infinicore::Tensor &kv_cache,
                                                                                           const infinicore::Tensor past_sequence_lengths) const {

    auto batch_size = key->size(0);
    auto update_len = key->size(2);
    auto k_cache_layer = kv_cache->narrow({{0, 0, 1}})->squeeze(0);
    auto v_cache_layer = kv_cache->narrow({{0, 1, 1}})->squeeze(0);

    size_t max_batch_size = k_cache_layer->size(0);
    size_t max_seq_len = k_cache_layer->size(2);
    auto device = k_cache_layer->device();

    ASSERT_EQ(batch_size, max_batch_size);

    size_t cache_pos = reinterpret_cast<int32_t *>(past_sequence_lengths->to(infinicore::Device{infinicore::Device::Type::kCpu})->data())[0];
    auto result_len = cache_pos + update_len;
    ASSERT(result_len <= max_seq_len);

    auto k_cache_update = k_cache_layer->narrow({{2, cache_pos, update_len}});
    auto v_cache_update = v_cache_layer->narrow({{2, cache_pos, update_len}});

    k_cache_update->copy_from(key);
    v_cache_update->copy_from(value);

    return {k_cache_layer, v_cache_layer};
}

} // namespace infinilm::layers::attention::backends
