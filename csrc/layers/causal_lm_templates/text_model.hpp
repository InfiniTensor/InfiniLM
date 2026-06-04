#pragma once

#include "../../config/model_config.hpp"
#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../utils.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/nn/embedding.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/tensor.hpp"
#include <memory>
#include <string>
#include <vector>

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
    TextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              const infinicore::Device &device)
        : device_(device),
          dtype_(model_config->get_dtype()) {
        vocab_size_ = model_config->get<size_t>("vocab_size");
        hidden_size_ = model_config->get<size_t>("hidden_size");
        size_t max_position_embeddings = model_config->get<size_t>("max_position_embeddings");
        size_t num_hidden_layers = model_config->get<size_t>("num_hidden_layers");
        double rope_theta = model_config->get<double>("rope_theta");
        double rms_norm_eps = model_config->get<double>("rms_norm_eps");

        embed_tokens_ = this->register_module<infinicore::nn::Embedding>("embed_tokens", vocab_size_, hidden_size_, std::nullopt, dtype_, device_);

        layers_.reserve(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            layers_.push_back(this->register_module<DecoderLayer>("layers." + std::to_string(i), model_config, i, device_));
        }

        norm_ = this->register_module<infinicore::nn::RMSNorm>("norm", hidden_size_, rms_norm_eps, dtype_, device_);

        this->_initialize_preallocated_workspace();
    }

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const {
        auto input_ids = input.input_ids.value();
        auto positions = input.position_ids.value();
        // 1. Embed tokens: input_ids -> [batch, seq_len, hidden_size]
        const auto shape = input_ids->shape();
        const size_t bs = shape[0];
        const size_t seq_len = shape[1];
        auto hidden_states = max_hidden_states_->narrow({{0, 0, bs * seq_len}})->view({bs, seq_len, hidden_size_});
        embed_tokens_->forward_(hidden_states, input_ids);

        // 2. Process through all decoder layers
        size_t num_layers = layers_.size();
        infinicore::Tensor residual;
        for (size_t i = 0; i < num_layers; ++i) {
            layers_.at(i)->forward(
                positions,
                hidden_states,
                residual);
        }

        norm_->forward_inplace(hidden_states, residual);
        return hidden_states;
    }

    infinicore::Tensor forward_naive(const infinilm::InfinilmModel::Input &input) const {
        // Don't use preallocated workspace in forward_naive function.
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

        // Don't use preallocated workspace in forward_embeds function.
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

private:
    void _initialize_preallocated_workspace() {
        const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
        auto &preallocated_workspace = infinilm::global_state::get_forward_context().preallocated_workspace;
        const size_t max_num_batched_tokens = infinilm_config.max_num_batched_tokens;

        const std::string text_model_cache_key = std::string("TextModel_max_num_batched_tokens_") + std::to_string(max_num_batched_tokens) + "_hidden_size_" + std::to_string(hidden_size_) + "_dtype_" + infinicore::toString(dtype_) + "_device_" + device_.toString();

        if (preallocated_workspace.find(text_model_cache_key) == preallocated_workspace.end()) {
            auto text_model_buffer = infinicore::Tensor::empty({max_num_batched_tokens, hidden_size_}, dtype_, device_);
            preallocated_workspace[text_model_cache_key] = text_model_buffer;
        }

        auto text_model_buffer = preallocated_workspace.at(text_model_cache_key);
        const auto text_model_buffer_shape = text_model_buffer->shape();
        ASSERT(text_model_buffer_shape[0] == max_num_batched_tokens && text_model_buffer_shape[1] == hidden_size_);

        max_hidden_states_ = text_model_buffer;
    }

protected:
    size_t vocab_size_;
    size_t hidden_size_;
    infinicore::Device device_;
    infinicore::DataType dtype_;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(DecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

private:
    // preallocated workspace for TextModel
    infinicore::Tensor max_hidden_states_;
};

} // namespace infinilm::layers::causal_lm_templates
