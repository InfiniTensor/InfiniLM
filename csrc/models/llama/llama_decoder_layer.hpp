#pragma once

#include "llama_config.hpp"
#include "llama_attention.hpp"
#include "llama_mlp.hpp"
#include "llama_hooks.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"

namespace infinilm::models::llama {

/**
 * @brief Single decoder layer (transformer block) for Llama
 *
 * Each decoder layer consists of:
 * - Input layer normalization (RMSNorm)
 * - Self-attention mechanism
 * - Post-attention layer normalization (RMSNorm)
 * - MLP feed-forward network
 *
 * Residual connections are applied around both attention and MLP blocks.
 */
class LlamaDecoderLayer : public infinicore::nn::Module {
public:
    /**
     * @brief Construct LlamaDecoderLayer module
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     */
    LlamaDecoderLayer(const LlamaConfig &config, const infinicore::Device &device);

    /**
     * @brief Forward pass: process one decoder layer
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache Optional KV cache for incremental decoding
     * @param hook_registry Optional hook registry for capturing intermediate values
     * @param hook_prefix Prefix for hook names (e.g., "layer0")
     * @param layer_idx Layer index for hooks
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

protected:
    // Layer normalization
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);

    // Attention and MLP
    INFINICORE_NN_MODULE(LlamaAttention, self_attn);
    INFINICORE_NN_MODULE(LlamaMLP, mlp);
};

} // namespace infinilm::models::llama
