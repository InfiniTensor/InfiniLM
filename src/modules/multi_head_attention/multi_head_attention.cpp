#include "multi_head_attention.hpp"

#include "../../context/inference_context.hpp"

#include <cmath>

namespace infinicore::nn::module {
std::shared_ptr<MultiHeadAttention> MultiHeadAttention::init(
    infiniDtype_t dtype,
    size_t hidden_size,
    size_t n_q_heads,
    size_t n_kv_heads,
    size_t qk_head_dim,
    size_t v_head_dim,
    int nranks,
    bool has_qkv_bias,
    bool has_qk_norm) {
    auto result = std::shared_ptr<MultiHeadAttention>(new MultiHeadAttention());

    size_t q_out_dim = n_q_heads * qk_head_dim;
    size_t k_out_dim = n_kv_heads * qk_head_dim;
    size_t v_out_dim = n_kv_heads * v_head_dim;

    result->q_proj = Linear::init(hidden_size, q_out_dim, dtype, has_qkv_bias, false, 
                                 weights::DistributionType::COLUMN, nranks);
    result->k_proj = Linear::init(hidden_size, k_out_dim, dtype, has_qkv_bias, false,
                                 weights::DistributionType::COLUMN, nranks);
    result->v_proj = Linear::init(hidden_size, v_out_dim, dtype, has_qkv_bias, false,
                                 weights::DistributionType::COLUMN, nranks);
    result->o_proj = Linear::init(q_out_dim, hidden_size, dtype, false, false,
                                 weights::DistributionType::ROW, nranks);

    if (has_qk_norm) {
        result->q_norm = RMSNorm::init(qk_head_dim, dtype);
        result->k_norm = RMSNorm::init(qk_head_dim, dtype);
    }

    return result;
}

void MultiHeadAttention::register_weights(infinicore::weights::Loader &loader, const std::string &name_prefix, int rank) {
    q_proj->register_weights(loader, name_prefix + ".q_proj", rank);
    k_proj->register_weights(loader, name_prefix + ".k_proj", rank);
    v_proj->register_weights(loader, name_prefix + ".v_proj", rank);
    o_proj->register_weights(loader, name_prefix + ".o_proj", rank);
    
    if (q_norm != nullptr) {
        q_norm->register_weights(loader, name_prefix + ".q_norm", rank);
    }
    if (k_norm != nullptr) {
        k_norm->register_weights(loader, name_prefix + ".k_norm", rank);
    }
}

void MultiHeadAttention::forward(std::shared_ptr<Tensor> output, 
                                std::shared_ptr<Tensor> input, 
                                std::shared_ptr<Tensor> residual,
                                std::shared_ptr<Tensor> attn_score_buf,
                                std::shared_ptr<Tensor> attn_val_buf,
                                std::shared_ptr<Tensor> pos_ids,
                                std::shared_ptr<Tensor> sin_table,
                                std::shared_ptr<Tensor> cos_table,
                                std::vector<std::shared_ptr<Tensor>> k_caches,
                                std::vector<std::shared_ptr<Tensor>> v_caches,
                                const std::vector<uint32_t>& req_lens,
                                const std::vector<uint32_t>& req_pos,
                                uint32_t nreq) {
    
    auto context = getInferenceContext();
    auto ntok = input->shape()[0];
    auto hidden_size = input->shape()[1];
    auto ngroup = n_q_heads / n_kv_heads;

    // Project Q, K, V
    auto q_buf = Tensor::buffer(input->dtype(), {ntok, n_q_heads * head_dim}, context.memory_pool);
    auto k_buf = Tensor::buffer(input->dtype(), {ntok, n_kv_heads * head_dim}, context.memory_pool);
    auto v_buf = Tensor::buffer(input->dtype(), {ntok, n_kv_heads * head_dim}, context.memory_pool);
    
    q_proj->forward(q_buf, input, nullptr);
    k_proj->forward(k_buf, input, nullptr);
    v_proj->forward(v_buf, input, nullptr);

    // Apply normalization if enabled
    if (q_norm != nullptr) {
        auto q_norm_temp = Tensor::buffer(input->dtype(), q_buf->shape(), context.memory_pool);
        q_norm->forward(q_norm_temp, q_buf);
        q_buf = q_norm_temp;
    }
    if (k_norm != nullptr) {
        auto k_norm_temp = Tensor::buffer(input->dtype(), k_buf->shape(), context.memory_pool);
        k_norm->forward(k_norm_temp, k_buf);
        k_buf = k_norm_temp;
    }

    // Reshape for attention computation
    auto q = q_buf->view({ntok, n_q_heads, head_dim});
    auto k = k_buf->view({ntok, n_kv_heads, head_dim});
    auto v = v_buf->view({ntok, n_kv_heads, head_dim});

    // Apply RoPE
    context.rope(q, q, pos_ids, sin_table, cos_table);
    context.rope(k, k, pos_ids, sin_table, cos_table);

    // Create attention output buffer
    auto attn_output_buf = Tensor::buffer(input->dtype(), {ntok, n_q_heads * head_dim}, context.memory_pool);

    // Process each request separately for attention computation
    size_t token_offset = 0;
    for (uint32_t req = 0; req < nreq; req++) {
        auto past_len = req_pos[req];
        auto seq_len = req_lens[req];
        auto total_len = past_len + seq_len;

        // Get slices for this request
        auto q_req = q->slice({{token_offset, token_offset + seq_len}});
        auto k_req = k->slice({{token_offset, token_offset + seq_len}});
        auto v_req = v->slice({{token_offset, token_offset + seq_len}});

        // Update KV caches for this request
        context.rearrange(k_caches[req]->slice({{past_len, past_len + seq_len}}), k_req);
        context.rearrange(v_caches[req]->slice({{past_len, past_len + seq_len}}), v_req);

        // Prepare attention computation
        auto q_reshaped = q_req->view({seq_len, n_kv_heads, ngroup, head_dim})->permute({1, 2, 0, 3});
        auto k_cached = k_caches[req]->slice({{0, total_len}})->permute({1, 2, 0});
        auto v_cached = v_caches[req]->slice({{0, total_len}})->permute({1, 0, 2});

        // Compute QK^T / sqrt(d)
        auto qk_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        auto qk_slice = attn_score_buf->slice(0, 0, n_q_heads * seq_len * total_len)
                            ->view({n_kv_heads, ngroup * seq_len, total_len});
        
        context.linear(qk_slice, q_reshaped->view({n_kv_heads, ngroup * seq_len, head_dim}), 
                      k_cached, qk_scale, 0.0f, nullptr, nullptr);

        // Apply causal softmax
        auto qk_softmax = qk_slice->view({n_q_heads, seq_len, total_len});
        context.causalSoftmax(qk_softmax, qk_softmax);

        // Compute attention values
        auto attn_val_slice = attn_val_buf->slice(0, 0, ngroup * seq_len)
                                 ->view({n_kv_heads, ngroup * seq_len, head_dim});
        
        context.linear(attn_val_slice, qk_slice, v_cached, 1.0f, 0.0f, nullptr, nullptr);

        // Store attention output (don't apply o_proj yet)
        auto attn_output_reshaped = attn_val_slice->view({n_kv_heads, ngroup, seq_len, head_dim})
                                       ->permute({2, 0, 1, 3})->view({seq_len, n_q_heads * head_dim});
        
        auto attn_output_slice = attn_output_buf->slice({{token_offset, token_offset + seq_len}});
        context.rearrange(attn_output_slice, attn_output_reshaped);

        token_offset += seq_len;
    }

    o_proj->forward(output, attn_output_buf, residual);
}

} // namespace infinicore::nn::module