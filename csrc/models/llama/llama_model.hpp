#pragma once

#include "llama_config.hpp"
#include "llama_decoder_layer.hpp"
#include "cache/kv_cache.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"
#include <vector>
#include <memory>

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
     * @param dtype Optional data type for model parameters (defaults to F32)
     */
    LlamaModel(const LlamaConfig &config, const infinicore::Device &device,
               infinicore::DataType dtype = infinicore::DataType::F32);

    /**
     * @brief Forward pass: process input through the model
     *
     * @param input_ids Token IDs tensor of shape [batch, seq_len]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache Optional model-level KV cache for incremental decoding
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &position_ids,
                                void *kv_cache = nullptr) const;


    // Module information
    const LlamaConfig &config() const { return config_; }
    size_t num_layers() const { return config_.num_hidden_layers; }

    /**
     * @brief Get the internal cache as an opaque pointer
     * @return Opaque pointer to the cache, or nullptr if cache hasn't been created yet
     */
    void *cache() const {
        return cache_ ? cache_.get() : nullptr;
    }

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
    // Persistent cache for when no external cache is provided
    // Mutable because it's not part of the model's learned parameters,
    // but needs to persist across forward calls for incremental decoding
    mutable std::unique_ptr<infinilm::cache::DynamicCache> cache_;
};

} // namespace infinilm::models::llama
