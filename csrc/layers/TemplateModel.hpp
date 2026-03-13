#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/model_config.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <vector>

#include "../engine/distributed/distributed.hpp"

namespace infinilm::models::layers {

/**
 * @brief Template model architecture (without language modeling head)
 *
 * A generic template class for transformer models that can work with any decoder layer type.
 * This is the core transformer model consisting of:
 * - Token embeddings (embed_tokens)
 * - Multiple decoder layers (layers)
 * - Final layer normalization (norm)
 * - Rotary Position Embeddings (rotary_emb)
 *
 * @tparam DecoderLayer The decoder layer type (e.g., TemplateDecoderLayer, Qwen3DecoderLayer)
 *
 * Usage example:
 * @code
 * using MyModel = TemplateModel<TemplateDecoderLayer<LlamaAttention, MLP>>;
 * MyModel model(config, device, rank_info);
 * @endcode
 */
template <typename DecoderLayer>
class TemplateModel : public infinicore::nn::Module {
public:
    /**
     * @brief Construct TemplateModel module
     *
     * @param model_config Model configuration
     * @param device Device to create tensors on
     * @param rank_info Rank information for distributed training
     * @param attention_backend Attention backend to use
     */
    TemplateModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device,
                  engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                  backends::AttentionBackend attention_backend = backends::AttentionBackend::Default) : model_config_(model_config), rank_info_(rank_info) {
        const auto &dtype{model_config_->get_dtype()};
        // Initialize token embeddings
        embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", model_config_->get<size_t>("vocab_size"), model_config_->get<size_t>("hidden_size"),
                                  std::nullopt, dtype, device);
        // Initialize decoder layers with layer indices
        // TODO: Update INFINICORE_NN_MODULE_VEC_INIT macro to support per-layer constructor arguments
        //       (e.g., via a factory function or lambda that receives the layer index)
        //       Currently, we can't use the macro because each layer needs a different layer_idx
        layers_.reserve(model_config_->get<size_t>("num_hidden_layers"));
        for (size_t i = 0; i < model_config_->get<size_t>("num_hidden_layers"); ++i) {
            layers_.push_back(this->register_module<DecoderLayer>(
                "layers." + std::to_string(i), model_config_, device, i, rank_info, attention_backend));
        }
        // Initialize final layer normalization
        norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                                  dtype, device);
        // Initialize Rotary Position Embeddings (shared across all layers)
        // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
        rotary_emb_ = this->register_module<infinicore::nn::RoPE>("rotary_emb", model_config_->get_head_dim(), model_config_->get<size_t>("max_position_embeddings"),
                                  model_config_->get<double>("rope_theta"), infinicore::nn::RoPE::Algo::GPT_NEOX,
                                  dtype, device, model_config_->get_rope_scaling());

        for (auto &layer : layers_) {
            if (layer) {
                layer->set_rotary_emb(rotary_emb_);
            }
        }
    }

    /**
     * @brief Forward pass: process input through the model
     *
     * @param input_ids Token IDs tensor of shape [batch, seq_len]. Batch is 1 when continuous batch is used,
     *                 and tokens from all requests are concatenated along seq_len dimension.
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_cache KV cache for incremental decoding
     * @param past_sequence_lengths Cache positions tensor of shape [n_req]
     * @param total_sequence_lengths Total sequence lengths tensor of shape [n_req]
     * @param input_offsets Input offsets (starting position) of each request in a continuous batch of shape [n_req + 1]
     * @param cu_seqlens Cumulative total sequence lengths for each request, of shape [n_req + 1]
     * @param block_tables Block ids for each request [batch, max_block_table_length]. Used for paged cache.
     * @param slot_mapping Slot ids for each token [seq]. Used for paged cache.
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                               const infinicore::Tensor &position_ids,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache,
                               std::optional<infinicore::Tensor> past_sequence_lengths,
                               std::optional<infinicore::Tensor> total_sequence_lengths,
                               std::optional<infinicore::Tensor> input_offsets,
                               std::optional<infinicore::Tensor> cu_seqlens,
                               std::optional<infinicore::Tensor> block_tables,
                               std::optional<infinicore::Tensor> slot_mapping) const {
        // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
        auto hidden_states = embed_tokens_->forward(input_ids);

        // 2. Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                hidden_states,
                residual,
                position_ids,
                kv_cache,
                past_sequence_lengths,
                total_sequence_lengths,
                input_offsets,
                cu_seqlens,
                block_tables,
                slot_mapping);
        }

        norm_->forward_inplace(hidden_states, residual);

        return hidden_states;
    }

    // Module information
    size_t num_layers() const { return model_config_->get<size_t>("num_hidden_layers"); }

protected:
    // Token embeddings
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);

    // Decoder layers
    INFINICORE_NN_MODULE_VEC(DecoderLayer, layers);

    // Final normalization
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    // Rotary Position Embeddings (shared across all layers)
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);

    engine::distributed::RankInfo rank_info_;

private:
    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

} // namespace infinilm::models::layers
