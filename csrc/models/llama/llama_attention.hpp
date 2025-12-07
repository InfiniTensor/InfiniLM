#pragma once

#include "cache/kv_cache.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include "llama_config.hpp"
#include <algorithm>
#include <memory>
#include <utility>

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::models::llama {

/**
 * @brief Multi-head self-attention module for Llama
 *
 * Implements the attention mechanism with:
 * - Query, Key, Value projections
 * - Output projection
 * - Rotary Position Embeddings (RoPE) applied to Q and K
 * - Support for Grouped Query Attention (GQA)
 */
class LlamaAttention : public infinicore::nn::Module {
public:
    /**
     * @brief Construct LlamaAttention module
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     * @param dtype Optional data type for model parameters (defaults to F32)
     */
    LlamaAttention(const LlamaConfig &config,
                   const infinicore::Device &device,
                   infinicore::DataType dtype = infinicore::DataType::F32,
                   engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    /**
     * @brief Forward pass: compute attention
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache Optional KV cache for incremental decoding
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids,
                               void *kv_cache = nullptr) const;

    /**
     * @brief Provide shared RoPE module from parent model.
     */
    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb);

    // Module information
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

protected:
    // Projection layers
    // INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
    // INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
    // INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
    // INFINICORE_NN_MODULE(infinicore::nn::Linear, o_proj);

    // Projection layers
    INFINICORE_NN_MODULE(infinicore::nn::ColumnParallelLinear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::ColumnParallelLinear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::ColumnParallelLinear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RowParallelLinear, o_proj);

    // Shared Rotary Position Embeddings (RoPE)
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

private:
    size_t hidden_size_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    size_t kv_dim_;
    bool use_bias_;

    // Internal KV cache for when no external cache is provided
    mutable infinilm::cache::KVCache internal_cache_;

    engine::distributed::RankInfo rank_info_;
};

} // namespace infinilm::models::llama