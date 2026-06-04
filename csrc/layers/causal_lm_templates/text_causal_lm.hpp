#pragma once

#include "../../global_state/global_state.hpp"
#include "../../models/infinilm_model.hpp"
#include "../../utils.hpp"
#include "../linear/linear.hpp"
#include "infinicore/device.hpp"
#include "infinicore/dtype.hpp"
#include <string>

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
                 const infinicore::Device &device)
        : device_(device),
          dtype_(model_config->get_dtype()) {
        model_config_ = model_config;

        size_t hidden_size = model_config->get<size_t>("hidden_size");
        vocab_size_ = model_config->get<size_t>("vocab_size");

        model_ = this->register_module<Model>("model", model_config, device);
        lm_head_ = this->register_module<infinilm::layers::linear::ReplicatedLinear>("lm_head", hidden_size, vocab_size_, false, dtype_, device_);

        this->_initialize_preallocated_workspace();
    }

    /**
     * @brief Forward pass: compute language modeling logits
     */
    Output forward(const Input &input) const override {
        auto hidden_states = model_->forward(input);

        const auto shape = hidden_states->shape();
        const size_t bs = shape[0];
        const size_t seq_len = shape[1];

        auto logits = max_logits_->narrow({{0, 0, bs * seq_len}})->view({bs, seq_len, vocab_size_});
        lm_head_->forward_(logits, hidden_states);
        return {logits};
    }

    infinicore::Tensor logits_from_hidden(const infinicore::Tensor &hidden_states) const {
        return lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
    }

    Model &model() { return *model_; }

protected:
    size_t vocab_size_;
    infinicore::Device device_;
    infinicore::DataType dtype_;

    INFINICORE_NN_MODULE(Model, model);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, lm_head);

private:
    void _initialize_preallocated_workspace() {
        const auto &infinilm_config = infinilm::global_state::get_infinilm_config();
        auto &preallocated_workspace = infinilm::global_state::get_forward_context().preallocated_workspace;
        const size_t max_num_batched_tokens = infinilm_config.max_num_batched_tokens;

        const std::string text_causal_lm_cache_key = std::string("TextCausalLM_max_num_batched_tokens_")
                                                   + std::to_string(max_num_batched_tokens) + "_vocab_size_"
                                                   + std::to_string(vocab_size_) + "_dtype_"
                                                   + infinicore::toString(dtype_) + "_device_"
                                                   + device_.toString();

        if (preallocated_workspace.find(text_causal_lm_cache_key) == preallocated_workspace.end()) {
            auto logits_buffer = infinicore::Tensor::empty({max_num_batched_tokens, vocab_size_}, dtype_, device_);
            preallocated_workspace[text_causal_lm_cache_key] = logits_buffer;
        }

        auto logits_buffer = preallocated_workspace.at(text_causal_lm_cache_key);
        const auto logits_buffer_shape = logits_buffer->shape();
        ASSERT(logits_buffer_shape[0] == max_num_batched_tokens && logits_buffer_shape[1] == vocab_size_);

        max_logits_ = logits_buffer;
    }

    // preallocated workspace for TextCausalLM
    infinicore::Tensor max_logits_;
};

} // namespace infinilm::layers::causal_lm_templates
