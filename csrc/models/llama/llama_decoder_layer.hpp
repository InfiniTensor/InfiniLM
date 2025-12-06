#pragma once

#include "llama_config.hpp"
#include "llama_attention.hpp"
#include "llama_mlp.hpp"
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
     * @param dtype Optional data type for model parameters (defaults to F32)
     */
    LlamaDecoderLayer(const LlamaConfig &config, const infinicore::Device &device,
                     infinicore::DataType dtype = infinicore::DataType::F32);

    /**
     * @brief Forward pass: process one decoder layer
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache Optional KV cache for incremental decoding
     * @param layer_idx Layer index for cache management and debugging
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                                const infinicore::Tensor &position_ids,
                                void *kv_cache = nullptr,
                                size_t layer_idx = -1) const;

    void set_rotary_emb(const std::shared_ptr<infinicore::nn::RoPE> &rotary_emb) {
        if (self_attn_) {
            self_attn_->set_rotary_emb(rotary_emb);
        }
    }


protected:
    // Layer normalization
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, input_layernorm);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, post_attention_layernorm);

    // Attention and MLP
    INFINICORE_NN_MODULE(LlamaAttention, self_attn);
    INFINICORE_NN_MODULE(LlamaMLP, mlp);
};

} // namespace infinilm::models::llama
