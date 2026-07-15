#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../linear/linear.hpp"
#include "infinicore/device.hpp"
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
        is_last_stage_ = infinilm::global_state::is_last_pipeline_stage();
        is_pipeline_parallel_ = infinilm::global_state::get_pipeline_model_parallel_world_size() > 1;
        if (is_last_stage_) {
            lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("lm_head", hidden_size, vocab_size, false, dtype, device);
        }
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input) const override {
        auto stage_output = model_->forward_stage(input);
        if (!is_last_stage_) {
            return {infinicore::Tensor{}, stage_output.hidden_states, stage_output.residual};
        }
        auto logits = lm_head_->forward(stage_output.hidden_states);
        if (!is_pipeline_parallel_) {
            return {logits};
        }
        return {logits, stage_output.hidden_states, stage_output.residual};
    }

    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const {
        if (!lm_head_) {
            throw std::runtime_error("lm_head is only available on the last pipeline stage");
        }
        return lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    }

    Model &model() { return *model_; }

protected:
    INFINICORE_NN_MODULE(Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);
    bool is_last_stage_{true};
    bool is_pipeline_parallel_{false};
};

} // namespace infinilm::layers::causal_lm_templates
