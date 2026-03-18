#include "static_attn.hpp"

#include "../../utils.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "infinicore/ops/mul.hpp"

#include "infinicore/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <optional>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

namespace infinilm::models::layers::attention {

StaticAttention::StaticAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                 const infinicore::Device &device,
                                 size_t layer_idx,
                                 engine::distributed::RankInfo rank_info)
    : AttentionBase(model_config, device, layer_idx, rank_info) {}

infinicore::Tensor StaticAttention::forward(const infinicore::Tensor &hidden_states,
                                            const infinilm::InfinilmModel::Input &input,
                                            std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
    if (!rotary_emb_) {
        throw std::runtime_error("StaticAttention: rotary_emb not configured");
    }

    if (auto paged_kv_cache = std::dynamic_pointer_cast<cache::PagedKVCache>(kv_cache)) {
        throw std::runtime_error("StaticAttention: paged_kv_cache not supported");
    }

    auto position_ids = input.position_ids.value();
    auto past_sequence_lengths = input.past_sequence_lengths;
    auto total_sequence_lengths = input.total_sequence_lengths;

    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto [q, k, v] = qkv_proj_->forward_split(hidden_states_mutable);

    if (qk_norm_) {
        q = q_norm_->forward(q->view({batch_size * seq_len, num_attention_heads_, head_dim_}));
        k = k_norm_->forward(k->view({batch_size * seq_len, num_key_value_heads_, head_dim_}));
    }

    // 2. Reshape for multi-head attention
    // Reshape Q, K, V to include batch dimension
    // Python: query_states = self.q_proj(hidden_states).view(querys_shape)
    // The view operation requires the tensor to be contiguous in the required dimensions
    auto q_reshaped = q->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    auto k_reshaped = k->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    auto v_reshaped = v->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 3. Prepare position_ids for RoPE - align with Python pattern
    // Python: bs, num = pos_ids.shape; pos_ids = pos_ids.view((bs * num,))
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;
    if (pos_shape.size() == 2) {
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    // 4. Apply RoPE to Q and K
    auto q_rope = infinicore::Tensor::empty({batch_size, num_attention_heads_, seq_len, head_dim_}, q_reshaped->dtype(), q_reshaped->device())->permute({0, 2, 1, 3});
    rotary_emb_->forward(q_rope, q_reshaped, pos_ids_for_rope); // [bs, seq_len, n_q_head, head_dim]
    rotary_emb_->forward(k_reshaped, pos_ids_for_rope, true);   // [bs, seq_len, n_kv_head, head_dim]

    // 5. Prepare KV caches
    // Convert to [batch, n_head, seq_len, head_dim] for cache
    // Ensure contiguous after permute for F16 compatibility with cache operations
    q_reshaped = q_rope->permute({0, 2, 1, 3});          // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3}); // [bs, n_kv_head, seq_len, head_dim]
    infinicore::Tensor k_total;                          // [bs, n_kv_head, max_seq_len, head_dim]
    infinicore::Tensor v_total;                          // [bs, n_kv_head, max_seq_len, head_dim]
    if (kv_cache == nullptr) {
        k_total = k_permuted;
        v_total = v_permuted;
    } else if (auto static_kv_cache = std::dynamic_pointer_cast<cache::StaticKVCache>(kv_cache)) {
        auto [k_total_tmp, v_total_tmp] = static_kv_cache->update(layer_idx_, k_permuted, v_permuted, past_sequence_lengths.value());
        k_total = k_total_tmp;
        v_total = v_total_tmp;
    } else {
        throw std::runtime_error("StaticAttention: Unsupported kvcache type");
    }

    infinicore::Tensor attn_output;
    if (false) {
        // experimental nineoothed flash attention
        attn_output = infinicore::op::flash_attention(q_reshaped, k_total, v_total, total_sequence_lengths.value(), scaling_, true);
        attn_output = attn_output->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_attention_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    } else {
        size_t total_seq_len = reinterpret_cast<int32_t *>(total_sequence_lengths.value()->to(infinicore::Device::cpu())->data())[0];
        k_total = k_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]
        v_total = v_total->narrow({{2, 0, total_seq_len}}); // [bs, n_kv_head, total_seq_len, head_dim]

        // 6. Compute attention
        size_t ngroup = num_attention_heads_ / num_key_value_heads_;
        auto Q = q_reshaped->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});
        auto K = k_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});
        auto V = v_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});

        auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

        auto attn_weight = infinicore::op::matmul(Q, K_transposed, scaling_); // [bs * n_kv_head, ng * seq_len, total_seq_len]

        auto attn_weight_softmax = attn_weight->view({batch_size * num_attention_heads_, seq_len, total_seq_len});
        infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

        auto out = infinicore::op::matmul(attn_weight, V); // [bs * n_kv_head, ng * seq_len, head_dim]

        attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                          ->permute({0, 2, 1, 3})
                          ->contiguous()
                          ->view({batch_size, seq_len, num_attention_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]
    }

    auto output = o_proj_->forward(attn_output);

    return output;
}

} // namespace infinilm::models::layers::attention
