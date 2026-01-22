#pragma once

#include "../../cache/kv_cache.hpp"
#include "../../engine/distributed/distributed.hpp"
#include "../../layers/fused_linear.hpp"
#include "llama_config.hpp"

#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include "llama_config.hpp"
#include <algorithm>
#include <memory>
#include <utility>

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
     * @param layer_idx Layer index for cache access
     * @param dtype Optional data type for model parameters (defaults to F32)
     */
    LlamaAttention(const LlamaConfig &config,
                   const infinicore::Device &device,
                   size_t layer_idx,
                   engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    /**
     * @brief Forward pass: compute attention
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache Optional model-level KV cache for incremental decoding
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &position_ids,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache,
                               std::optional<infinicore::Tensor> past_sequence_lengths,
                               std::optional<infinicore::Tensor> total_sequence_lengths,
                               std::optional<infinicore::Tensor> input_offsets,
                               std::optional<infinicore::Tensor> block_tables,
                               std::optional<infinicore::Tensor> slot_mapping) const;

    /**
     * @brief Get the layer index
     */
    size_t layer_idx() const { return layer_idx_; }

    /**
     * @brief Provide shared RoPE module from parent model.
     */
    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb);

    // Module information
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

private:
    infinicore::Tensor forward_(const infinicore::Tensor &hidden_states,
                                const infinicore::Tensor &position_ids,
                                std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                std::optional<infinicore::Tensor> past_sequence_lengths,
                                std::optional<infinicore::Tensor> total_sequence_lengths) const;

    infinicore::Tensor forward_paged_(const infinicore::Tensor &hidden_states,
                                      const infinicore::Tensor &position_ids,
                                      std::shared_ptr<infinilm::cache::PagedKVCache> kv_cache,
                                      std::optional<infinicore::Tensor> total_sequence_lengths,
                                      std::optional<infinicore::Tensor> input_offsets,
                                      std::optional<infinicore::Tensor> block_tables,
                                      std::optional<infinicore::Tensor> slot_mapping) const;

protected:
    // Projection layers
    INFINICORE_NN_MODULE(infinilm::layers::QKVParallelLinear, qkv_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RowParallelLinear, o_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, q_norm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, k_norm);
    engine::distributed::RankInfo rank_info_;

    // Shared Rotary Position Embeddings (RoPE)
    std::shared_ptr<infinicore::nn::RoPE> rotary_emb_;

private:
    size_t layer_idx_; // Layer index for cache access
    size_t hidden_size_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    size_t kv_dim_;
    bool use_bias_;                  // Bias for Q/K/V projections
    bool use_output_bias_;           // Bias for output projection (o_proj)
    bool use_qk_norm_;               // Whether to use QK RMSNorm
    size_t max_position_embeddings_; // For cache initialization (deprecated, kept for compatibility)

    float scaling_;
};

} // namespace infinilm::models::llama
