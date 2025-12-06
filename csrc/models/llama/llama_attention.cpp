#include "llama_attention.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/mul.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <spdlog/spdlog.h>
#include <stdexcept>

namespace infinilm::models::llama {

LlamaAttention::LlamaAttention(const LlamaConfig &config, const infinicore::Device &device,
                               infinicore::DataType dtype)
    : hidden_size_(config.hidden_size),
      num_attention_heads_(config.num_attention_heads),
      num_key_value_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      kv_dim_(config.kv_dim()),
      use_bias_(config.attention_bias),
      use_output_bias_(config.attention_output_bias),
      max_position_embeddings_(config.max_position_embeddings) {
    // Initialize projection layers
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, hidden_size_, use_bias_,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, kv_dim_, use_bias_,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, kv_dim_, use_bias_,
                               dtype, device);
    // Output projection uses attention_output_bias (can be different from qkv)
    INFINICORE_NN_MODULE_INIT(o_proj, hidden_size_, hidden_size_, use_output_bias_,
                              dtype, device);
}

infinicore::Tensor LlamaAttention::forward(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &position_ids,
                                            void *kv_cache,
                                            size_t layer_idx) const {
    if (!rotary_emb_) {
        throw std::runtime_error("LlamaAttention: rotary_emb not configured");
    }
    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto q = q_proj_->forward(hidden_states_mutable); // [batch, seq_len, hidden_size]

    auto k = k_proj_->forward(hidden_states_mutable); // [batch, seq_len, kv_dim]

    auto v = v_proj_->forward(hidden_states_mutable); // [batch, seq_len, kv_dim]

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

    // 4. Prepare KV caches
    // Convert to [batch, n_head, seq_len, head_dim] for cache
    // Ensure contiguous after permute for F16 compatibility with cache operations
    q_reshaped = q_reshaped->permute({0, 2, 1, 3})->contiguous(); // [bs, n_q_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3});          // [bs, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3});          // [bs, n_kv_head, seq_len, head_dim]
    infinilm::cache::DynamicCache *external_cache = static_cast<infinilm::cache::DynamicCache *>(kv_cache);
    infinicore::Tensor k_total; // [bs, n_kv_head, total_seq_len, head_dim]
    infinicore::Tensor v_total; // [bs, n_kv_head, total_seq_len, head_dim]
    if (external_cache != nullptr) {
        auto [k_total_tmp, v_total_tmp] = external_cache->update(layer_idx, k_permuted, v_permuted);
        k_total = k_total_tmp;
        v_total = v_total_tmp;
    } else {
        // No external cache - this shouldn't happen in normal operation, but handle gracefully
        throw std::runtime_error("LlamaAttention: kv_cache is required but nullptr provided");
    }
    auto total_seq_len = k_total->shape()[2];

    // 5. Apply RoPE to full batch
    auto q_rope = q_reshaped->view({batch_size * num_attention_heads_, seq_len, head_dim_})->permute({1, 0, 2});                                               // [seq_len, bs * n_q_head, head_dim]
    auto k_rope = k_total->narrow({{2, total_seq_len - seq_len, seq_len}})->view({batch_size * num_key_value_heads_, seq_len, head_dim_})->permute({1, 0, 2}); // [seq_len, bs * n_kv_head, head_dim]
    rotary_emb_->forward(q_rope, pos_ids_for_rope, true);
    rotary_emb_->forward(k_rope, pos_ids_for_rope, true);

    // 6. Compute attention
    size_t ngroup = num_attention_heads_ / num_key_value_heads_;
    auto Q = q_reshaped->view({batch_size * num_key_value_heads_, ngroup * seq_len, head_dim_});
    auto K = k_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});
    auto V = v_total->view({batch_size * num_key_value_heads_, total_seq_len, head_dim_});

    auto K_transposed = K->permute({0, 2, 1}); // [bs * n_kv_head, head_dim, total_seq_len]

    float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    auto attn_weight = infinicore::op::matmul(Q, K_transposed, scaling); // [bs * n_kv_head, ng * seq_len, total_seq_len]

    auto attn_weight_softmax = attn_weight->view({batch_size * num_attention_heads_, seq_len, total_seq_len});
    infinicore::op::causal_softmax_(attn_weight_softmax, attn_weight_softmax);

    auto out = infinicore::op::matmul(attn_weight, V); // [bs * n_kv_head, ng * seq_len, head_dim]

    auto attn_output = out->view({batch_size, num_attention_heads_, seq_len, head_dim_})
                           ->permute({0, 2, 1, 3})
                           ->contiguous()
                           ->view({batch_size, seq_len, num_attention_heads_ * head_dim_}); // [bs, seq_len, n_q_head * head_dim]

    auto output = o_proj_->forward(attn_output);

    return output;
}

void LlamaAttention::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    rotary_emb_ = rotary_emb;
}

} // namespace infinilm::models::llama
