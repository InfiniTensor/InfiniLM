#pragma once

#include "llama_config.hpp"
#include "llama_decoder_layer.hpp"
#include "llama_hooks.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"
#include <vector>

namespace infinilm::models::llama {

/**
 * @brief Main Llama model architecture (without language modeling head)
 *
 * This is the core transformer model consisting of:
 * - Token embeddings (embed_tokens)
 * - Multiple decoder layers (layers)
 * - Final layer normalization (norm)
 * - Rotary Position Embeddings (rotary_emb)
 *
 * This matches the structure of HuggingFace's LlamaModel.
 */
class LlamaModel : public infinicore::nn::Module {
public:
    /**
     * @brief Construct LlamaModel module
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     */
    LlamaModel(const LlamaConfig &config, const infinicore::Device &device);

    /**
     * @brief Forward pass: process input through the model
     *
     * @param input_ids Token IDs tensor of shape [batch, seq_len]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_caches Optional KV caches for incremental decoding (one per layer)
     * @param hook_registry Optional hook registry for capturing intermediate values
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     *
     * Note: This is a placeholder forward method. The actual implementation
     * will be added when integrating with the inference engine.
     */
    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &position_ids,
                                std::vector<void *> *kv_caches = nullptr,
                                const HookRegistry *hook_registry = nullptr) const;

    // Module information
    const LlamaConfig &config() const { return config_; }
    size_t num_layers() const { return config_.num_hidden_layers; }

protected:
    // Token embeddings
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);

    // Decoder layers
    INFINICORE_NN_MODULE_VEC(LlamaDecoderLayer, layers);

    // Final normalization
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    // Rotary Position Embeddings (shared across all layers)
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);

private:
    LlamaConfig config_;
};

} // namespace infinilm::models::llama
