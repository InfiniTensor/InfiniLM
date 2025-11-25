#pragma once

#include "llama_model.hpp"
#include "llama_hooks.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"

namespace infinilm::models::llama {

/**
 * @brief Llama model for Causal Language Modeling
 *
 * Extends LlamaModel by adding a language modeling head (lm_head) that
 * projects hidden states to vocabulary logits.
 *
 * This matches the structure of HuggingFace's LlamaForCausalLM.
 */
class LlamaForCausalLM : public infinicore::nn::Module {
public:
    /**
     * @brief Construct LlamaForCausalLM module
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     */
    LlamaForCausalLM(const LlamaConfig &config, const infinicore::Device &device);

    /**
     * @brief Forward pass: compute language modeling logits
     *
     * @param input_ids Token IDs tensor of shape [batch, seq_len]
     * @param position_ids Position IDs tensor of shape [batch, seq_len] or [seq_len]
     * @param kv_caches Optional KV caches for incremental decoding (one per layer)
     * @param hook_registry Optional hook registry for capturing intermediate values
     * @return Logits tensor of shape [batch, seq_len, vocab_size]
     *
     * Note: This is a placeholder forward method. The actual implementation
     * will be added when integrating with the inference engine.
     */
    infinicore::Tensor forward(const infinicore::Tensor &input_ids,
                                const infinicore::Tensor &position_ids,
                                std::vector<void *> *kv_caches = nullptr,
                                const HookRegistry *hook_registry = nullptr) const;

    // Module information
    const LlamaConfig &config() const { return model_->config(); }
    LlamaModel &model() { return *model_; }
    const LlamaModel &model() const { return *model_; }

protected:
    // Base model
    INFINICORE_NN_MODULE(LlamaModel, model);

    // Language modeling head
    INFINICORE_NN_MODULE(infinicore::nn::Linear, lm_head);
};

} // namespace infinilm::models::llama
