#include "llama_for_causal_lm.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
#include <iostream>

namespace infinilm::models::llama {

LlamaForCausalLM::LlamaForCausalLM(const LlamaConfig &config,
                                   const infinicore::Device &device,
                                   engine::distributed::RankInfo rank_info) {

    // Initialize module's device_ member
    device_ = device;

    const auto &dtype{config.dtype};

    // Initialize base model
    INFINICORE_NN_MODULE_INIT(model, config, device, rank_info);

    // Initialize language modeling head
    // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
    // For now, we create a separate linear layer
    INFINICORE_NN_MODULE_INIT(lm_head, config.hidden_size, config.vocab_size, false,
                              dtype, device);
}

LlamaForCausalLM::Output LlamaForCausalLM::forward(const Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();
    auto cache_lengths = input.cache_lengths;
    auto input_lengths = input.input_lengths;
    auto input_offsets = input.input_offsets;
    auto block_tables = input.block_tables;
    auto slot_mapping = input.slot_mapping;
    auto random_val = input.random_val;
    auto topp = input.topp;
    auto topk = input.topk;
    auto temperature = input.temperature;

    // 1. Forward through base model to get hidden states
    auto position_ids_device = position_ids->to(device_);
    auto hidden_states = model_->forward(input_ids, position_ids_device, cache_lengths, input_lengths, input_offsets, block_tables, slot_mapping);

    // 2. Apply language modeling head to get logits
    auto logits = lm_head_->forward(hidden_states);

    // 3. Perform random sampling
    const auto &logits_shape{logits->shape()};
    const auto &batch_size{logits_shape[0]};
    const auto &vocab_size{logits_shape[2]};

    auto output_ids{infinicore::Tensor::empty({batch_size}, infinicore::DataType::I32, device_)};

    for (auto i{decltype(batch_size)(0)}; i < batch_size; ++i) {
        auto score{logits->narrow({{0, i, 1}})->view({vocab_size})};
        auto out{output_ids->narrow({{0, i, 1}})->view({})};
        infinicore::op::random_sample_(
            out, score, random_val, topp, topk, temperature);
    }

    // 4. Synchronize
    if (device_.getType() != infinicore::Device::Type::CPU) {
        infinicore::context::syncStream();
    }

    return {output_ids};
}

void LlamaForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    model_->reset_cache(cache_config);
}

} // namespace infinilm::models::llama
