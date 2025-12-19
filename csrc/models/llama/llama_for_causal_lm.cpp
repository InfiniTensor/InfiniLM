#include "llama_for_causal_lm.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama {

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig &config,
                                   const infinicore::Device &device,
                                   infinicore::DataType dtype,
                                   engine::distributed::RankInfo rank_info) {

    // Initialize module's device_ member
    device_ = device;

    // Initialize base model
    INFINICORE_NN_MODULE_INIT(model, config, device, dtype, rank_info);

    // Initialize language modeling head
    // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
    // For now, we create a separate linear layer
    INFINICORE_NN_MODULE_INIT(lm_head, config.hidden_size, config.vocab_size, false,
                              dtype, device);
}

LlamaForCausalLM::Output LlamaForCausalLM::forward(const Input &input) const {
    const auto &[input_ids, position_ids, kv_cache] = input;

    // 1. Forward through base model to get hidden states
    auto position_ids_device = position_ids->to(device_);
    auto hidden_states = model_->forward(input_ids, position_ids_device, kv_cache);

    // 2. Apply language modeling head to get logits
    auto logits = lm_head_->forward(hidden_states);

    return {logits};
}

void LlamaForCausalLM::reset_cache(size_t pos) {
    model_->reset_cache(pos);
}

void LlamaForCausalLM::reset_cache(const cache::CacheConfig &new_config, size_t pos) {
    model_->reset_cache(new_config, pos);
}

} // namespace infinilm::models::llama
