#include "llama_for_causal_lm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig &config, const infinicore::Device &device,
                                   infinicore::DataType dtype) {
    // Initialize base model
    INFINICORE_NN_MODULE_INIT(model, config, device, dtype);

    // Initialize language modeling head
    // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
    // For now, we create a separate linear layer
    INFINICORE_NN_MODULE_INIT(lm_head, config.hidden_size, config.vocab_size, false,
                              dtype, device);
}

infinicore::Tensor LlamaForCausalLM::forward(const infinicore::Tensor &input_ids,
                                              const infinicore::Tensor &position_ids,
                                              std::vector<void *> *kv_caches) const {
    // 1. Forward through base model to get hidden states
    auto hidden_states = model_->forward(input_ids, position_ids, kv_caches);

    // 2. Apply language modeling head to get logits
    auto logits = lm_head_->forward(hidden_states);

    return logits;
}

} // namespace infinilm::models::llama
