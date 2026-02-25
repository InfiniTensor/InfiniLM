#include "llama_for_causal_lm.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
namespace infinilm::models::llama {
/**
 * @deprecated This function is deprecated and will be REMOVED in the next major release (v0.2.0).
 *
 * ⚠️ DEVELOPMENT POLICY:
 *   - NO new development or feature additions permitted on this interface
 *   - Only critical bug fixes (security/stability) allowed until removal
 *   - All new code MUST migrate to the polymorphic overload below
 *
 * Replacement: Use the polymorphic overload of this same function name with updated signature
 * Reason: Legacy signature lacks support for dynamic quantization modes.
 * Removal target: v0.2.0 (Q2 2026)
 */
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

LlamaForCausalLM::LlamaForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device,
                                   engine::distributed::RankInfo rank_info) {

    // Initialize module's device_ member
    device_ = device;
    const auto &dtype{model_config->get_dtype()};

    // Initialize base model
    INFINICORE_NN_MODULE_INIT(model, model_config, device, rank_info);
    // Initialize language modeling head
    // Note: If tie_word_embeddings is true, we would share weights with embed_tokens
    // For now, we create a separate linear layer

    INFINICORE_NN_MODULE_INIT(lm_head, model_config->get<size_t>("hidden_size"), model_config->get<size_t>("vocab_size"), false,
                              dtype, device);
}

LlamaForCausalLM::Output LlamaForCausalLM::forward(const Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();
    auto past_sequence_lengths = input.past_sequence_lengths;
    auto total_sequence_length = input.total_sequence_lengths;
    auto input_offsets = input.input_offsets;
    auto block_tables = input.block_tables;
    auto slot_mapping = input.slot_mapping;
    auto max_sequence_length = input.max_sequence_length;

    // 1. Forward through base model to get hidden states
    auto hidden_states = model_->forward(
        input_ids, position_ids, past_sequence_lengths, total_sequence_length, input_offsets, block_tables, slot_mapping, max_sequence_length);

    // 2. Apply language modeling head to get logits
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void LlamaForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    cache_config_ = cache_config->unique_copy();
    model_->reset_cache(cache_config_.get());
}

const cache::CacheConfig *LlamaForCausalLM::get_cache_config() const {
    return cache_config_.get();
}

} // namespace infinilm::models::llama
