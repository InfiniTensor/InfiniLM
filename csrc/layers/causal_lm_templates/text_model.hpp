#pragma once

#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <stdexcept>

namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Text model architecture (without language modeling head).
 *
 * Generic transformer model consisting of:
 * - Token embeddings
 * - Multiple decoder layers
 * - Final layer normalization
 *
 * @tparam DecoderLayer The decoder layer type (e.g., Qwen3DecoderLayer)
 */
template <typename DecoderLayer>
class TextModel : public infinicore::nn::Module {
public:
    struct StageOutput {
        infinicore::Tensor hidden_states;
        infinicore::Tensor residual;
    };

    TextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device) {
        const auto &dtype{model_config->get_dtype()};
        size_t vocab_size = model_config->get<size_t>("vocab_size");
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        const auto layer_range = infinilm::global_state::get_pipeline_layer_range(num_hidden_layers);
        layer_start_ = layer_range.start;
        layer_end_ = layer_range.end;
        is_first_stage_ = infinilm::global_state::is_first_pipeline_stage();
        is_last_stage_ = infinilm::global_state::is_last_pipeline_stage();

        if (is_first_stage_) {
            embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", vocab_size, hidden_size, std::nullopt, dtype, device);
        }

        layers_.reserve(layer_end_ - layer_start_);
        for (size_t i = layer_start_; i < layer_end_; ++i) {
            layers_.push_back(this->register_module<DecoderLayer>("layers." + std::to_string(i), model_config, i, device));
        }

        if (is_last_stage_) {
            norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size, rms_norm_eps, dtype, device);
        }
    }

    StageOutput forward_stage(const infinilm::InfinilmModel::Input &input) const {
        auto positions = input.position_ids.value();
        infinicore::Tensor hidden_states;
        infinicore::Tensor residual;
        if (is_first_stage_) {
            if (!input.input_ids.has_value()) {
                throw std::invalid_argument("First pipeline stage requires input_ids");
            }
            hidden_states = embed_tokens_->forward(input.input_ids.value());
        } else {
            if (!input.pp_hidden_states.has_value() || !input.pp_residual.has_value()) {
                throw std::invalid_argument("Non-first pipeline stage requires hidden_states and residual");
            }
            hidden_states = input.pp_hidden_states.value();
            residual = input.pp_residual.value();
        }

        for (const auto &layer : layers_) {
            layer->forward(
                positions,
                hidden_states,
                residual);
        }

        if (is_last_stage_) {
            norm_->forward_inplace(hidden_states, residual);
        }
        return {hidden_states, residual};
    }

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const {
        return forward_stage(input).hidden_states;
    }

    infinicore::Tensor forward_naive(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        auto positions = input.position_ids.value();
        auto hidden_states = embed_tokens_->forward(input_ids);
        size_t num_layers = layers_.size();
        for (size_t i = 0; i < num_layers; ++i) {
            hidden_states = layers_.at(i)->forward(positions, hidden_states);
        }
        hidden_states = norm_->forward(hidden_states);
        return hidden_states;
    }

    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds,
                                      const infinicore::Tensor &position_ids) const {

        auto hidden_states = inputs_embeds;

        //  Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                position_ids,
                hidden_states,
                residual);
        }

        norm_->forward_inplace(hidden_states, residual);
        return hidden_states;
    }

    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const {
        return embed_tokens_->forward(input_ids);
    }

protected:
    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    size_t layer_start_{0};
    size_t layer_end_{0};
    bool is_first_stage_{true};
    bool is_last_stage_{true};
};

} // namespace infinilm::layers::causal_lm_templates
