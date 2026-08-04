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
        const auto &rank_info = infinilm::global_state::get_tensor_model_parallel_rank_info();
        pp_size_ = static_cast<size_t>(rank_info.pp_size);
        pp_stage_ = static_cast<size_t>(rank_info.pp_stage);

        model_ = this->register_module<Model>("model", model_config, device);
        if (is_last_pp_stage()) {
            lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("lm_head", hidden_size, vocab_size, false, dtype, device);
        }
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input) const override {
        auto hidden_states = model_->forward(input);
        if (!is_last_pp_stage()) {
            return {infinicore::Tensor(), hidden_states};
        }
        auto logits = lm_head_->forward(hidden_states);
        return {logits, hidden_states};
    }

    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const {
        if (!lm_head_) {
            throw std::runtime_error("TextCausalLM::logits_from_hidden called on a non-last pipeline stage");
        }
        return lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    }

    Model &model() { return *model_; }

protected:
    INFINICORE_NN_MODULE(Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);

private:
    bool is_last_pp_stage() const { return pp_stage_ + 1 == pp_size_; }

    size_t pp_size_{1};
    size_t pp_stage_{0};
};

} // namespace infinilm::layers::causal_lm_templates
