#pragma once

#include "../../cache/kv_cache.hpp"
#include "llama_config.hpp"
#include "llama_decoder_layer.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include "llama_config.hpp"
#include "llama_decoder_layer.hpp"
#include <memory>
#include <vector>

#include "../../engine/distributed/distributed.hpp"

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
    LlamaModel(const LlamaConfig &config,
               const infinicore::Device &device,
               engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    /**
     * @brief Forward pass: process input through the model
     *
     * @param input_ids Token IDs tensor of shape [batch, seq_len]. Batch is 1 when continuous batch is used,
     *                 and tokens from all requests are concatenated along seq_len dimension.
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param past_sequence_lengths Cache positions tensor of shape [n_req]
     * @param total_sequence_lengths Total sequence lengths tensor of shape [n_req]
     * @param input_offsets Input offsets (starting position) of each request in a continuous batch of shape [n_req + 1]
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &position_ids,
                               std::optional<infinicore::Tensor> past_sequence_lengths,
                               std::optional<infinicore::Tensor> total_sequence_lengths,
                               std::optional<infinicore::Tensor> input_offsets,
                               std::optional<infinicore::Tensor> block_tables,
                               std::optional<infinicore::Tensor> slot_mapping) const;

    void reset_cache(const cache::CacheConfig *cache_config);

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

    engine::distributed::RankInfo rank_info_;

    std::shared_ptr<cache::Cache> kv_cache_;

private:
    LlamaConfig config_;
};

} // namespace infinilm::models::llama
