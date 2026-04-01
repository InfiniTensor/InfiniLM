#include "static_attn.hpp"
#include "../../../global_state/global_state.hpp"
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
                                                std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                                                const infinilm::global_state::AttentionMetadata &attn_metadata) const {

    auto k_scale = layer.get_k_scale();
    auto v_scale = layer.get_v_scale();
    if (infinicore::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
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

    // update static kv cache
    // k_total:  [bs, n_kv_head, max_seq_len, head_dim]
    // v_total : [bs, n_kv_head, max_seq_len, head_dim]
    auto [k_total, v_total] = do_kv_cache_update(layer, k_permuted, v_permuted, kv_cache, past_sequence_lengths.value());

    infinicore::Tensor attn_output;
    if (false) {
        // experimental nineoothed flash attention
        attn_output = infinicore::op::flash_attention(q_reshaped, k_total, v_total, total_sequence_lengths.value(), scale_, true);
        attn_output = attn_output->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    } else {
        size_t total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];

        if (infinicore::quantization::KVQuantAlgo::NONE != this->kv_quant_scheme_) {
            infinilm::KVQuantUtils::dequantize(
                k_total, v_total,
                this->kv_quant_scheme_,
                k_scale,
                v_scale,
                q_reshaped);
        }

        k_total = k_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]
        v_total = v_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]

        //  Compute attention
        size_t ngroup = num_heads_ / num_kv_heads_;
        auto Q = q_reshaped->view({batch_size * num_kv_heads_, ngroup * seq_len, head_dim_});
        auto K = k_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});
        auto V = v_total->view({batch_size * num_kv_heads_, total_seq_len, head_dim_});

        auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

        auto attn_weight = infinicore::op::matmul(Q, K_transposed, scale_); // [bs * n_kv_head, ng * seq_len, total_seq_len]

        auto attn_weight_softmax = attn_weight->view({batch_size * num_heads_, seq_len, total_seq_len});
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

        auto out = infinicore::op::matmul(attn_weight, V); // [bs * n_kv_head, ng * seq_len, head_dim]

        attn_output = out->view({batch_size, num_heads_, seq_len, head_dim_})
                          ->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    }
    return attn_output;
}

std::tuple<infinicore::Tensor, infinicore::Tensor> StaticAttentionImpl::do_kv_cache_update(const AttentionLayer &layer,
                                                                                           const infinicore::Tensor key,
                                                                                           const infinicore::Tensor value,
                                                                                           std::tuple<infinicore::Tensor, infinicore::Tensor> kv_cache,
                                                                                           const infinicore::Tensor past_sequence_lengths) const {
    auto batch_size = key->size(0);
    auto update_len = key->size(2);
    auto k_cache_layer = std::get<0>(kv_cache);
    auto v_cache_layer = std::get<1>(kv_cache);

    size_t max_batch_size = k_cache_layer->size(0);
    size_t max_seq_len = k_cache_layer->size(2);
    auto device = k_cache_layer->device();

    ASSERT_EQ(batch_size, max_batch_size);

    size_t cache_pos = reinterpret_cast<int32_t *>(past_sequence_lengths->to(infinicore::Device::cpu())->data())[0];
    auto result_len = cache_pos + update_len;
    ASSERT(result_len <= max_seq_len);

    auto k_cache_update = k_cache_layer->narrow({{2, cache_pos, update_len}});
    auto v_cache_update = v_cache_layer->narrow({{2, cache_pos, update_len}});

    k_cache_update->copy_from(key);
    v_cache_update->copy_from(value);

    return {k_cache_layer, v_cache_layer};
}

} // namespace infinilm::layers::attention::backends
