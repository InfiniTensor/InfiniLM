#pragma once

#include "../models/infinilm_model.hpp"
#include "infinicore/device.hpp"

namespace infinilm::layers {

/**
 * @brief Text Causal Language Modeling class
 *
 * A generic template class for Causal Language Modeling that can work with any model type.
 * This class extends any model (specified via template parameter) by adding a language
 * modeling head (lm_head) that projects hidden states to vocabulary logits.
 *
 * @tparam Model The base model type (e.g., Qwen3Model, LlamaModel, Qwen3MoeModel)
 *
 * Usage example:
 * @code
 * using MyCausalLM = TextCausalLM<Qwen3Model>;
 * MyCausalLM model(config, device );
 * @endcode
 */
template <typename Model>
class TextCausalLM : public InfinilmModel {
public:
    /**
     * @brief Construct TextCausalLM module
     *
     * @param model_config Model configuration
     * @param device Device to create tensors on
     * @param attention_backend Attention backend to use
     */
    TextCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device) {

        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t vocab_size = model_config->get<size_t>("vocab_size");

        const auto &dtype{model_config->get_dtype()};

        model_ = this->register_module<Model>("model", model_config, device);
        lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("lm_head", hidden_size, vocab_size, false, dtype, device);
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input) const override {
        auto hidden_states = model_->forward(input);
        auto logits = lm_head_->forward(hidden_states);
        return {logits};
    }

protected:
    INFINICORE_NN_MODULE(Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

} // namespace infinilm::layers
