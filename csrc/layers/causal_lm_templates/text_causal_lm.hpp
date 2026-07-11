#pragma once

#include "../../models/infinilm_model.hpp"
#include "../linear/linear.hpp"
#include "infinicore/device.hpp"
#include "infinicore/ops/cat.hpp"
#include <stdexcept>

namespace infinilm::layers::causal_lm_templates {

/**
 * @brief Text Causal Language Modeling class
 *
 * A generic template class for Causal Language Modeling.
 *
 * @tparam Model The base model type (e.g., Qwen3Model, Qwen3MoeModel)
 *
 * Usage example:
 * @code
 * using Qwen3CausalLM = TextCausalLM<Qwen3Model>;
 * @endcode
 */
template <typename Model>
class TextCausalLM : public InfinilmModel {
public:
    /**
     * @brief Construct TextCausalLM module
     *
     * @param model_config: Model configuration.
     * @param device: Device to create tensors on
     */
    TextCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                 const infinicore::Device &device) {
        model_config_ = model_config;

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
        auto lm_head_input = hidden_states;
        // Ordinary prefill needs only the final-position logits.
        if (!input.sample_all_positions
            && input.last_token_indices.has_value()
            && !input.last_token_indices->empty()
            && hidden_states->size(0) == 1
            && hidden_states->size(1) > input.last_token_indices->size()) {
            const auto &indices = input.last_token_indices.value();
            for (const size_t index : indices) {
                if (index >= hidden_states->size(1)) {
                    throw std::out_of_range("last-token index exceeds packed hidden states");
                }
            }
            if (indices.size() == 1) {
                lm_head_input = hidden_states->narrow({{1, indices.front(), 1}});
            } else {
                std::vector<infinicore::Tensor> final_hidden_states;
                final_hidden_states.reserve(indices.size());
                for (const size_t index : indices) {
                    final_hidden_states.push_back(
                        hidden_states->narrow({{1, index, 1}}));
                }
                lm_head_input = infinicore::op::cat(std::move(final_hidden_states), 1);
            }
        }
        auto logits = lm_head_->forward(lm_head_input);
        return {logits};
    }

    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const {
        return lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    }

    Model &model() { return *model_; }

protected:
    INFINICORE_NN_MODULE(Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
};

} // namespace infinilm::layers::causal_lm_templates
