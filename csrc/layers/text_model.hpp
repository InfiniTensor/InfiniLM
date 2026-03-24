#pragma once

#include "../config/model_config.hpp"
#include "../models/infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/nn/rope.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <vector>

namespace infinilm::layers {

/**
 * @brief Text model architecture (without language modeling head)
 *
 * Generic transformer model consisting of:
 * - Token embeddings
 * - Multiple decoder layers
 * - Final layer normalization
 * - Rotary Position Embeddings
 *
 * @tparam DecoderLayer The decoder layer type (e.g., TextDecoderLayer, Qwen3DecoderLayer)
 */
template <typename DecoderLayer>
class TextModel : public infinicore::nn::Module {
public:
    TextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device) {
        const auto &dtype{model_config->get_dtype()};

        size_t vocab_size = model_config->get<size_t>("vocab_size");
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        double rope_theta = model_config->get<double>("rope_theta");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", vocab_size, hidden_size, std::nullopt, dtype, device);

        layers_.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(this->register_module<DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
        }

        norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size, rms_norm_eps, dtype, device);

        rotary_emb_ = this->register_module<infinicore::nn::RoPE>(
            "rotary_emb", model_config->get_head_dim(), max_position_embeddings,
            rope_theta, infinicore::nn::RoPE::Algo::GPT_NEOX, dtype, device,
            model_config->get_rope_scaling());

        for (auto &layer : layers_) {
            if (layer) {
                layer->set_rotary_emb(rotary_emb_);
            }
        }
    }

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
        auto hidden_states = embed_tokens_->forward(input_ids);

        // 2. Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                hidden_states,
                residual);
        }

        norm_->forward_inplace(hidden_states, residual);

        return hidden_states;
    }

    infinicore::Tensor forward_naive(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        auto hidden_states = embed_tokens_->forward(input_ids);

        size_t num_layers = layers_.size();
        for (size_t i = 0; i < num_layers; ++i) {
            hidden_states = layers_.at(i)->forward(hidden_states);
        }

        hidden_states = norm_->forward(hidden_states);
        return hidden_states;
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    INFINICORE_NN_MODULE(infinicore::nn::RoPE, rotary_emb);
};

} // namespace infinilm::layers
