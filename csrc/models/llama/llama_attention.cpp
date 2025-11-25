#include "llama_attention.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/ops.hpp"
#include <spdlog/spdlog.h>

namespace infinilm::models::llama {

LlamaAttention::LlamaAttention(const LlamaConfig &config, const infinicore::Device &device)
    : hidden_size_(config.hidden_size),
      num_attention_heads_(config.num_attention_heads),
      num_key_value_heads_(config.num_key_value_heads),
      head_dim_(config.head_dim),
      kv_dim_(config.kv_dim()),
      use_bias_(config.attention_bias) {
    // Initialize projection layers
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, hidden_size_, use_bias_,
                               infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, kv_dim_, use_bias_,
                               infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, kv_dim_, use_bias_,
                               infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(o_proj, hidden_size_, hidden_size_, use_bias_,
                               infinicore::DataType::F32, device);

    // Initialize Rotary Position Embeddings
    // Use GPT_J-style inverse frequencies (default in InfiniCore) and GPT_NEOX rotation pairing
    INFINICORE_NN_MODULE_INIT(rotary_emb, head_dim_, config.max_position_embeddings,
                              config.rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX,
                              infinicore::DataType::F32, device);
}

infinicore::Tensor LlamaAttention::forward(const infinicore::Tensor &hidden_states,
                                            const infinicore::Tensor &position_ids,
                                            void *kv_cache,
                                            const HookRegistry *hook_registry,
                                            const std::string &hook_prefix,
                                            int layer_idx) const {
    // Input shape: [batch, seq_len, hidden_size]
    auto hidden_states_mutable = hidden_states;
    auto shape = hidden_states->shape();
    size_t batch_size = shape[0];
    size_t seq_len = shape[1];

    // 1. Project Q, K, V
    auto q = q_proj_->forward(hidden_states_mutable);  // [batch, seq_len, hidden_size]
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_q_after_proj", q, layer_idx);
    }

    auto k = k_proj_->forward(hidden_states_mutable);  // [batch, seq_len, kv_dim]
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_k_after_proj", k, layer_idx);
    }

    auto v = v_proj_->forward(hidden_states_mutable);  // [batch, seq_len, kv_dim]
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_v_after_proj", v, layer_idx);
    }

    // 2. Reshape for multi-head attention
    // Q: [batch, seq_len, hidden_size] -> [batch, seq_len, n_q_head, head_dim] -> [batch, n_q_head, seq_len, head_dim] -> [n_q_head, seq_len, head_dim]
    // K: [batch, seq_len, kv_dim] -> [batch, seq_len, n_kv_head, head_dim] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]
    // V: [batch, seq_len, kv_dim] -> [batch, seq_len, n_kv_head, head_dim] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]

    // Make tensors contiguous for reshaping
    auto q_cont = q->contiguous();
    auto k_cont = k->contiguous();
    auto v_cont = v->contiguous();

    // Reshape Q: [batch, seq_len, hidden_size] -> [batch, seq_len, n_q_head, head_dim] -> [batch, n_q_head, seq_len, head_dim] -> [n_q_head, seq_len, head_dim]
    // Validate dimensions before view
    size_t q_total_elements = q_cont->shape()[0] * q_cont->shape()[1] * q_cont->shape()[2];
    size_t q_expected_elements = batch_size * seq_len * num_attention_heads_ * head_dim_;
    if (q_total_elements != q_expected_elements) {
        SPDLOG_ERROR("LlamaAttention::forward: Dimension mismatch for Q reshape!");
        SPDLOG_ERROR("  Current total elements: {}", q_total_elements);
        SPDLOG_ERROR("  Expected total elements: {}", q_expected_elements);
        SPDLOG_ERROR("  Current shape: [{}, {}, {}]", q_cont->shape()[0], q_cont->shape()[1], q_cont->shape()[2]);
        SPDLOG_ERROR("  Target shape: [{}, {}, {}, {}]", batch_size, seq_len, num_attention_heads_, head_dim_);
        throw std::runtime_error("Dimension mismatch in Q reshape");
    }

    auto q_reshaped = q_cont->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    // Permute to [batch, n_q_head, seq_len, head_dim]
    auto q_permuted = q_reshaped->permute({0, 2, 1, 3});
    // For batch=1 (common in inference), reshape to [n_q_head, seq_len, head_dim]
    // Note: For batch > 1, this would need to be handled differently
    // Make contiguous before final view since permute can make tensor non-contiguous
    auto q_permuted_cont = q_permuted->contiguous();
    auto q_attn = q_permuted_cont->view({num_attention_heads_, seq_len, head_dim_});
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_q_after_reshape", q_attn, layer_idx);
    }

    // Reshape K: [batch, seq_len, kv_dim] -> [batch, seq_len, num_key_value_heads_, head_dim_] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]
    auto k_reshaped = k_cont->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    // Permute to [batch, n_kv_head, seq_len, head_dim]
    auto k_permuted = k_reshaped->permute({0, 2, 1, 3});
    // Make contiguous before final view
    auto k_permuted_cont = k_permuted->contiguous();
    // Reshape to [n_kv_head, seq_len, head_dim]
    auto k_attn = k_permuted_cont->view({num_key_value_heads_, seq_len, head_dim_});
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_k_after_reshape", k_attn, layer_idx);
    }

    // Reshape V: [batch, seq_len, kv_dim] -> [batch, seq_len, num_key_value_heads_, head_dim_] -> [batch, n_kv_head, seq_len, head_dim] -> [n_kv_head, seq_len, head_dim]
    auto v_reshaped = v_cont->view({batch_size, seq_len, num_key_value_heads_, head_dim_});
    // Permute to [batch, n_kv_head, seq_len, head_dim]
    auto v_permuted = v_reshaped->permute({0, 2, 1, 3});
    // Make contiguous before final view
    auto v_permuted_cont = v_permuted->contiguous();
    // Reshape to [n_kv_head, seq_len, head_dim]
    auto v_attn = v_permuted_cont->view({num_key_value_heads_, seq_len, head_dim_});
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_v_after_reshape", v_attn, layer_idx);
    }

    // 3. Prepare position_ids for RoPE
    // RoPE expects position_ids to be 1D [seq_len], but we receive [batch, seq_len]
    // Extract the first row and make it 1D, ensure it's contiguous
    auto pos_shape = position_ids->shape();
    infinicore::Tensor pos_ids_for_rope = position_ids;  // Initialize with position_ids (fallback for 1D case)
    if (pos_shape.size() == 2) {
        // Extract first row: narrow dimension 0, start=0, length=1
        auto pos_narrowed = position_ids->narrow({{0, 0, 1}});
        // Make contiguous and view as 1D
        pos_ids_for_rope = pos_narrowed->contiguous()->view({pos_shape[1]});
    } else if (pos_shape.size() == 1) {
        // Ensure it's contiguous
        pos_ids_for_rope = position_ids->contiguous();
    } else {
        throw std::runtime_error("Unexpected position_ids shape");
    }

    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_position_ids", pos_ids_for_rope, layer_idx);
    }

    // 4. Apply RoPE to Q and K
    // RoPE expects [seq_len, n_head, head_dim], but we have [n_head, seq_len, head_dim]
    // Permute to [seq_len, n_head, head_dim] before RoPE, then permute back
    // Permute from [n_head, seq_len, head_dim] to [seq_len, n_head, head_dim]
    auto q_for_rope = q_attn->permute({1, 0, 2});  // [seq_len, n_head, head_dim]
    // Make contiguous - RoPE kernel requires last dimension to be contiguous (stride(2) == 1)
    auto q_for_rope_cont = q_for_rope->contiguous();
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_q_before_rope", q_for_rope_cont, layer_idx);
    }
    auto q_rope_out = rotary_emb_->forward(q_for_rope_cont, pos_ids_for_rope);
    // Make RoPE output contiguous before permute - permute can cause issues with non-contiguous tensors
    auto q_rope_out_cont = q_rope_out->contiguous();
    // Permute back from [seq_len, n_head, head_dim] to [n_head, seq_len, head_dim]
    auto q_rope_permuted = q_rope_out_cont->permute({1, 0, 2});  // [n_head, seq_len, head_dim]
    // Make permuted tensor contiguous - permute creates non-contiguous tensors which can cause issues
    auto q_rope = q_rope_permuted->contiguous();
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_q_after_rope", q_rope, layer_idx);
        // Also capture the output directly from RoPE before permute
        hook_registry->call_hook(hook_prefix + "_q_rope_out_before_permute", q_rope_out, layer_idx);
    }

    // Permute from [n_kv_head, seq_len, head_dim] to [seq_len, n_kv_head, head_dim]
    auto k_for_rope = k_attn->permute({1, 0, 2});  // [seq_len, n_kv_head, head_dim]
    // Make contiguous - RoPE kernel requires last dimension to be contiguous (stride(2) == 1)
    auto k_for_rope_cont = k_for_rope->contiguous();
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_k_before_rope", k_for_rope_cont, layer_idx);
    }
    auto k_rope_out = rotary_emb_->forward(k_for_rope_cont, pos_ids_for_rope);
    // Make RoPE output contiguous before permute - permute can cause issues with non-contiguous tensors
    auto k_rope_out_cont = k_rope_out->contiguous();
    // Permute back from [seq_len, n_kv_head, head_dim] to [n_kv_head, seq_len, head_dim]
    auto k_rope_permuted = k_rope_out_cont->permute({1, 0, 2});  // [n_kv_head, seq_len, head_dim]
    // Make permuted tensor contiguous - permute creates non-contiguous tensors which can cause issues
    auto k_rope = k_rope_permuted->contiguous();
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_k_after_rope", k_rope, layer_idx);
        // Also capture the output directly from RoPE before permute
        hook_registry->call_hook(hook_prefix + "_k_rope_out_before_permute", k_rope_out, layer_idx);
    }

    // 5. Prepare KV caches for attention operation
    // The attention operation requires cache tensors with at least seq_len capacity
    // For first pass (pos=0), we create caches with seq_len capacity
    size_t cache_capacity = seq_len;  // Cache capacity (at least seq_len for first pass)
    auto k_cache = infinicore::Tensor::empty({num_key_value_heads_, cache_capacity, head_dim_},
                                             k_rope->dtype(), k_rope->device());
    auto v_cache = infinicore::Tensor::empty({num_key_value_heads_, cache_capacity, head_dim_},
                                             v_attn->dtype(), v_attn->device());

    // 6. Call attention operation
    // attention expects: q [n_q_head, seq_len, head_dim], k [n_kv_head, seq_len, head_dim], v [n_kv_head, seq_len, head_dim]
    // Returns: [seq_len, n_q_head, head_dim]
    // Note: V doesn't get RoPE applied, so we use v_attn directly
    size_t pos = 0;  // Position in cache (0 for first pass)
    auto attn_output = infinicore::op::attention(q_rope, k_rope, v_attn, k_cache, v_cache, pos);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_attention_output", attn_output, layer_idx);
    }

    // 7. Reshape output back: [seq_len, n_q_head, head_dim] -> [batch, seq_len, hidden_size]
    // attention returns [seq_len, n_q_head, head_dim]
    // Reshape to [n_q_head, seq_len, head_dim] -> [batch, n_q_head, seq_len, head_dim] -> [batch, seq_len, n_q_head, head_dim] -> [batch, seq_len, hidden_size]
    auto attn_cont = attn_output->contiguous();
    // Permute from [seq_len, n_q_head, head_dim] to [n_q_head, seq_len, head_dim]
    auto attn_permuted = attn_cont->permute({1, 0, 2});  // [n_q_head, seq_len, head_dim]
    // Reshape to [batch, n_q_head, seq_len, head_dim]
    // Make contiguous before view since permute can make tensor non-contiguous
    auto attn_permuted_cont = attn_permuted->contiguous();
    auto attn_batch = attn_permuted_cont->view({batch_size, num_attention_heads_, seq_len, head_dim_});
    // Permute to [batch, seq_len, n_q_head, head_dim]
    auto attn_final = attn_batch->permute({0, 2, 1, 3});
    // Reshape to [batch, seq_len, hidden_size]
    // Make contiguous before view since permute can make tensor non-contiguous
    auto attn_final_cont = attn_final->contiguous();
    auto attn_flat = attn_final_cont->view({batch_size, seq_len, hidden_size_});

    // 8. Apply output projection
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_attn_flat_before_o_proj", attn_flat, layer_idx);
    }
    auto output = o_proj_->forward(attn_flat);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_output", output, layer_idx);
    }
    return output;
}

infinicore::Tensor LlamaAttention::project_q(const infinicore::Tensor &hidden_states) const {
    auto mutable_hidden = hidden_states;
    return q_proj_->forward(mutable_hidden);
}

} // namespace infinilm::models::llama
