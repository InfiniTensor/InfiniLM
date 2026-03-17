#pragma once

#include "../backends/attention_backends.hpp"
#include "../cache/kv_cache.hpp"
#include "../config/model_config.hpp"
#include "../models/infinilm_model.hpp"

#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <cstddef>
#include <memory>
#include <stdexcept>
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

        size_t vocab_size = model_config_->get<size_t>("vocab_size");
        size_t hidden_size = model_config_->get<size_t>("hidden_size");
        size_t max_position_embeddings = model_config_->get<size_t>("max_position_embeddings");
        double rope_theta = model_config_->get<double>("rope_theta");
        double rms_norm_eps = model_config_->get<double>("rms_norm_eps");

        // Initialize token embeddings
        embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", vocab_size, hidden_size, std::nullopt, dtype, device);

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
        norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size, rms_norm_eps, dtype, device);
        // Initialize Rotary Position Embeddings (shared across all layers)
        // Use GPT-J-style inverse frequencies (default) and GPT_NEOX rotation pairing
        rotary_emb_ = this->register_module<infinicore::nn::RoPE>("rotary_emb", model_config_->get_head_dim(), max_position_embeddings,
                                                                  rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX,
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
     * @param kv_cache KV cache for incremental decoding
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input,
                               std::shared_ptr<infinilm::cache::Cache> kv_cache) const {
        auto input_ids = input.input_ids.value();

        // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
        auto hidden_states = embed_tokens_->forward(input_ids);

        // 2. Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                hidden_states,
                residual,
                input,
                kv_cache);
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
