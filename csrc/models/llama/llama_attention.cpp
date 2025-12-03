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
      use_bias_(config.attention_bias) {
    // Initialize projection layers
    INFINICORE_NN_MODULE_INIT(q_proj, hidden_size_, hidden_size_, use_bias_,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(k_proj, hidden_size_, kv_dim_, use_bias_,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(v_proj, hidden_size_, kv_dim_, use_bias_,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(o_proj, hidden_size_, hidden_size_, use_bias_,
                              dtype, device);
}

infinicore::Tensor LlamaAttention::forward(const infinicore::Tensor &hidden_states,
                                           const infinicore::Tensor &position_ids,
                                           void *kv_cache) const {
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

    // 4. Apply RoPE to full batch - align with Python pattern

    // Python: x = x.view((bs * seq_len, num_heads, head_dim))
    // Python asserts: seq_len * x_stride[1] == x_stride[0] (contiguous in dim=0 and dim=1)
    // The kernel requires stride(2) == 1 (last dimension contiguous)
    // Python's assertion + stride(2) == 1 means the tensor is fully contiguous
    // However, to be safe and match Python's behavior exactly, ensure fully contiguous
    auto q_for_rope = q_reshaped->view({batch_size * seq_len, num_attention_heads_, head_dim_})->contiguous();
    auto k_for_rope = k_reshaped->view({batch_size * seq_len, num_key_value_heads_, head_dim_})->contiguous();

    // Call RoPE on full batch (matching Python pattern)
    auto q_rope_out = rotary_emb_->forward(q_for_rope, pos_ids_for_rope);
    auto k_rope_out = rotary_emb_->forward(k_for_rope, pos_ids_for_rope);

    // Reshape back to [batch_size, seq_len, num_heads, head_dim] (matching Python pattern)
    q_rope_out = q_rope_out->view({batch_size, seq_len, num_attention_heads_, head_dim_});
    k_rope_out = k_rope_out->view({batch_size, seq_len, num_key_value_heads_, head_dim_});

    // 5. Process each batch item separately for attention computation
    infinilm::cache::KVCache *external_cache = static_cast<infinilm::cache::KVCache *>(kv_cache);
    auto output_tensor = infinicore::Tensor::empty(
        {batch_size, seq_len, hidden_size_},
        q->dtype(),
        q->device());

    for (size_t b = 0; b < batch_size; ++b) {
        // Extract batch item from RoPE output (already computed above for full batch)
        // Ensure contiguous after narrow+view to avoid stride issues in GEMM operations
        auto q_batch = q_rope_out->narrow({{0, b, 1}})->view({seq_len, num_attention_heads_, head_dim_});
        auto k_batch = k_rope_out->narrow({{0, b, 1}})->view({seq_len, num_key_value_heads_, head_dim_});
        auto v_batch = v_reshaped->narrow({{0, b, 1}})->view({seq_len, num_key_value_heads_, head_dim_});

        // Convert to [n_head, seq_len, head_dim] for cache
        // Ensure contiguous after permute for F16 compatibility with cache operations
        auto q_rope = q_batch->permute({1, 0, 2})->contiguous();     // [n_q_head, seq_len, head_dim]
        auto k_rope = k_batch->permute({1, 0, 2})->contiguous();     // [n_kv_head, seq_len, head_dim]
        auto v_permuted = v_batch->permute({1, 0, 2})->contiguous(); // [n_kv_head, seq_len, head_dim]

        // 5. Prepare KV caches
        infinicore::Tensor k_total;
        infinicore::Tensor v_total;
        if (external_cache != nullptr) {
            auto [k_total_tmp, v_total_tmp] = external_cache->update(k_rope, v_permuted);
            k_total = k_total_tmp;
            v_total = v_total_tmp;
        } else {
            auto [k_total_tmp, v_total_tmp] = internal_cache_.update(k_rope, v_permuted);
            k_total = k_total_tmp;
            v_total = v_total_tmp;
        }

        // 6. Compute attention - strictly align with Python pattern
        // Python: query_states_i = query_states.narrow(0, i, 1).view((seq_len, num_attention_heads, head_dim))
        // Python: key_states_i = key_states_total.narrow(0, i, 1).view((total_seq_len, num_key_value_heads, head_dim))
        // Python: value_states_i = value_states_total.narrow(0, i, 1).view((total_seq_len, num_key_value_heads, head_dim))
        // Python: attention_i = grouped_query_attention(query_states_i, key_states_i, value_states_i, scaling=self.scaling)

        // Extract from KV cache (k_total and v_total are [n_kv_head, total_seq_len, head_dim])
        // Python: key_states_total.narrow(0, i, 1).view((total_seq_len, num_key_value_heads, head_dim))
        // Python's narrow+view ensures contiguous memory, so we need to ensure contiguous before permute
        auto k_for_attn = k_total->permute({1, 0, 2}); // [total_seq_len, n_kv_head, head_dim]
        auto v_for_attn = v_total->permute({1, 0, 2}); // [total_seq_len, n_kv_head, head_dim]

        // q_batch is already [seq_len, n_q_head, head_dim] from above
        auto q_for_attn = q_batch; // [seq_len, n_q_head, head_dim]

        // Python: grouped_query_attention calls repeat_kv if ngroup > 1
        // Python: repeat_kv expands [total_seq_len, num_key_value_heads, head_dim] -> [total_seq_len, num_attention_heads, head_dim]
        size_t ngroup = num_attention_heads_ / num_key_value_heads_;
        if (ngroup > 1) {
            // Python: repeat_kv uses as_strided to expand
            size_t total_seq_len = k_for_attn->shape()[0];
            size_t n_kv_head = k_for_attn->shape()[1];
            size_t head_dim = k_for_attn->shape()[2];

            auto k_strides = k_for_attn->strides();
            auto k_strided = k_for_attn->as_strided(
                {total_seq_len, n_kv_head, ngroup, head_dim},
                {k_strides[0], k_strides[1], 0, k_strides[2]});
            k_for_attn = k_strided->contiguous()->view({total_seq_len, n_kv_head * ngroup, head_dim});

            auto v_strides = v_for_attn->strides();
            auto v_strided = v_for_attn->as_strided(
                {total_seq_len, n_kv_head, ngroup, head_dim},
                {v_strides[0], v_strides[1], 0, v_strides[2]});
            v_for_attn = v_strided->contiguous()->view({total_seq_len, n_kv_head * ngroup, head_dim});
        }

        // Python: multi_head_attention(querys, keys, values, scaling)
        // Python: Q = querys.permute((1, 0, 2))  # [num_heads, seq_len, head_dim]
        // Python: K = keys  # [total_seq_len, num_heads, head_dim] (NO permute!)
        // Python: V = values.permute((1, 0, 2))  # [num_heads, total_seq_len, head_dim]
        auto Q = q_for_attn->permute({1, 0, 2}); // [n_q_head, seq_len, head_dim]
        auto K = k_for_attn;                     // [total_seq_len, n_q_head, head_dim] - keep as-is (matching Python)
        auto V = v_for_attn->permute({1, 0, 2}); // [n_q_head, total_seq_len, head_dim]

        // Python: attn_weight = Q @ K.permute((1, 2, 0))
        // Python: K.permute((1, 2, 0)) transforms [total_seq_len, num_heads, head_dim] -> [num_heads, head_dim, total_seq_len]
        auto K_transposed = K->permute({1, 2, 0}); // [n_q_head, head_dim, total_seq_len]

        // Use GEMM with alpha=scaling to combine scaling with matrix multiplication
        // This is more efficient than doing matmul followed by mul
        float scaling = 1.0f / std::sqrt(static_cast<float>(head_dim_));
        auto attn_weight = infinicore::op::matmul(Q, K_transposed, scaling); // [n_q_head, seq_len, total_seq_len]

        infinicore::op::causal_softmax_(attn_weight, attn_weight);

        auto out = infinicore::op::matmul(attn_weight, V); // [n_q_head, seq_len, head_dim]

        // Python: return out.permute((1, 0, 2)).contiguous()  # [seq_len, num_heads, head_dim]
        auto attn_output = out->permute({1, 0, 2})->contiguous(); // [seq_len, n_q_head, head_dim]

        // Python: attn_output_i.copy_(attention_i)
        // Python: attn_output = attn_output.view(hidden_states_shape)  # [bs, seq_len, hidden_size]
        // Copy to output tensor - attn_output is [seq_len, num_attention_heads, head_dim]
        auto output_batch = output_tensor->narrow({{0, b, 1}})->view({seq_len, hidden_size_});
        auto attn_flat = attn_output->contiguous()->view({seq_len, hidden_size_});
        output_batch->copy_from(attn_flat);
    }

    // 8. Apply output projection to all batches
    auto output = o_proj_->forward(output_tensor);

    return output;
}

void LlamaAttention::set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
    rotary_emb_ = rotary_emb;
}

} // namespace infinilm::models::llama
