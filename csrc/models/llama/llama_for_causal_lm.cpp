#include "llama_for_causal_lm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig &config, const infinicore::Device &device) {
    // Initialize base model
    INFINICORE_NN_MODULE_INIT(model, config, device);

    // Initialize language modeling head
    // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
    // For now, we create a separate linear layer
    INFINICORE_NN_MODULE_INIT(lm_head, config.hidden_size, config.vocab_size, false,
                              infinicore::DataType::F32, device);
}

infinicore::Tensor LlamaForCausalLM::forward(const infinicore::Tensor &input_ids,
                                              const infinicore::Tensor &position_ids,
                                              std::vector<void *> *kv_caches,
                                              const HookRegistry *hook_registry) const {
    // 1. Forward through base model to get hidden states
    auto hidden_states = model_->forward(input_ids, position_ids, kv_caches, hook_registry);

    // 2. Apply language modeling head to get logits
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook("hidden_states_before_lm_head", hidden_states, -1);
    }
    auto logits = lm_head_->forward(hidden_states);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook("logits", logits, -1);
    }

    return logits;
}

} // namespace infinilm::models::llama
