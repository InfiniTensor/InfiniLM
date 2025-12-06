#include "llama_for_causal_lm.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/ops/random_sample.hpp"
#include "infinicore/context/context.hpp"
#include <iostream>

namespace infinilm::models::llama {

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig &config, const infinicore::Device &device,
                                   infinicore::DataType dtype) {

    // Initialize module's device_ member
    device_ = device;

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
                                             void *kv_cache) const {
    // 1. Forward through base model to get hidden states
    auto position_ids_device = position_ids->to(device_);
    auto hidden_states = model_->forward(input_ids, position_ids_device, kv_cache);

    // 2. Apply language modeling head to get logits
    auto logits = lm_head_->forward(hidden_states);

    // 3. CRITICAL: Synchronize the C++ backend's context after forward pass
    // This ensures all C++ backend operations complete before returning to Python
    if (device_.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::setDevice(device_, false);
        infinicore::context::syncStream();
    }

    return logits;
}

infinicore::Tensor LlamaForCausalLM::forward(std::vector<std::any> args) const {
    if (args.size() < 2) {
        throw std::invalid_argument("LlamaForCausalLM::forward requires at least 2 arguments: input_ids and position_ids");
    }

    // Extract input tensors from args
    const auto &input_ids = std::any_cast<const infinicore::Tensor &>(args[0]);
    const auto &position_ids = std::any_cast<const infinicore::Tensor &>(args[1]);

    // Optional KV caches
    std::vector<void *> *kv_caches = nullptr;
    if (args.size() >= 3) {
        kv_caches = std::any_cast<std::vector<void *> *>(args[2]);
    }

    return forward(input_ids, position_ids, kv_caches);
}

} // namespace infinilm::models::llama
