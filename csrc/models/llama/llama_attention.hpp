#pragma once

#include "llama_config.hpp"
#include "llama_hooks.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"

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
     */
    LlamaAttention(const LlamaConfig &config, const infinicore::Device &device);

    /**
     * @brief Forward pass: compute attention
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache Optional KV cache for incremental decoding
     * @param hook_registry Optional hook registry for capturing intermediate values
     * @param hook_prefix Prefix for hook names (e.g., "layer0_attention")
     * @param layer_idx Layer index for hooks (-1 if not applicable)
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     *
     * Note: This is a placeholder forward method. The actual implementation
     * will be added when integrating with the inference engine.
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                                const infinicore::Tensor &position_ids,
                                void *kv_cache = nullptr,
                                const HookRegistry *hook_registry = nullptr,
                                const std::string &hook_prefix = "",
                                int layer_idx = -1) const;

    /**
     * @brief Minimal helper to run only the Q projection.
     *
     * This is primarily intended for reproducing low-level runtime issues
     * (e.g., allocator failures) without executing the full attention block.
     */
    infinicore::Tensor project_q(const infinicore::Tensor &hidden_states) const;

    // Module information
    size_t num_heads() const { return num_attention_heads_; }
    size_t num_kv_heads() const { return num_key_value_heads_; }
    size_t head_dim() const { return head_dim_; }
    size_t hidden_size() const { return hidden_size_; }

protected:
    // Projection layers
    INFINICORE_NN_MODULE(infinicore::nn::Linear, q_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, k_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, v_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, o_proj);

    // Rotary Position Embeddings (RoPE)
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);

private:
    size_t hidden_size_;
    size_t num_attention_heads_;
    size_t num_key_value_heads_;
    size_t head_dim_;
    size_t kv_dim_;
    bool use_bias_;
};

} // namespace infinilm::models::llama
