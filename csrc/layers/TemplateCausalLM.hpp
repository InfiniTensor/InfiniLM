#pragma once

#include "../backends/attention_backends.hpp"
#include "../models/infinilm_model.hpp"

#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../engine/distributed/distributed.hpp"

namespace infinilm::layers {

/**
 * @brief Template Causal Language Modeling class
 *
 * A generic template class for Causal Language Modeling that can work with any model type.
 * This class extends any model (specified via template parameter) by adding a language
 * modeling head (lm_head) that projects hidden states to vocabulary logits.
 *
 * @tparam Model The base model type (e.g., Qwen3Model, LlamaModel, Qwen3MoeModel)
 *
 * Usage example:
 * @code
 * using MyCausalLM = TemplateCausalLM<Qwen3Model>;
 * MyCausalLM model(config, device, rank_info);
 * @endcode
 */
template <typename Model>
class TemplateCausalLM : public InfinilmModel {
public:
    /**
     * @brief Construct TemplateCausalLM module
     *
     * @param model_config Model configuration
     * @param device Device to create tensors on
     * @param rank_info Rank information for distributed training
     * @param attention_backend Attention backend to use
     */
    TemplateCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device,
                     engine::distributed::RankInfo rank_info = engine::distributed::RankInfo(),
                     backends::AttentionBackend attention_backend = backends::AttentionBackend::Default) {
        // Initialize base class members
        attention_backend_ = attention_backend;
        model_config_ = model_config;
        rank_info_ = rank_info;
        size_t hidden_size = model_config->get<size_t>("hidden_size");
        size_t vocab_size = model_config->get<size_t>("vocab_size");

        // Initialize module's device_ member
        device_ = device;
        const auto &dtype{model_config->get_dtype()};

        // Initialize base model
        model_ = this->register_module<Model>("model", model_config, device, rank_info, attention_backend);
        // Initialize language modeling head
        // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
        // For now, we create a separate linear layer
        lm_head_ = this->register_module<infinicore::nn::Linear>("lm_head", hidden_size, vocab_size, false, dtype, device);
    }

    /**
     * @brief Forward pass: compute language modeling logits
     *
     * @param input Encapsulated input tensors and other parameters
     * @return Output structure containing the result
     */
    Output forward(const Input &input) const {
        // 1. Forward through base model to get hidden states
        auto hidden_states = model_->forward(input, kv_cache_);

        // 2. Apply language modeling head to get logits
        auto logits = lm_head_->forward(hidden_states);
        return {logits};
    }

    // Module information
    Model &model() { return *model_; }
    const Model &model() const { return *model_; }
    size_t num_layers() const { return model_->num_layers(); }

protected:
    // Base model
    INFINICORE_NN_MODULE(Model, model);

    // Language modeling head
    INFINICORE_NN_MODULE(infinicore::nn::Linear, lm_head);
};

} // namespace infinilm::layers
