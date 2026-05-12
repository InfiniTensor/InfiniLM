#include "llama_for_causal_lm.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/ops.hpp"
namespace infinilm::models::llama_legacy {

LlamaForCausalLM::LlamaForCausalLM(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                   const infinicore::Device &device,
                                   engine::distributed::RankInfo rank_info,
                                   backends::AttentionBackend attention_backend) {
    spdlog::warn("infinilm::models::llama_legacy: LlamaForCausalLM is no longer supported, please use the new model instead.");

    device_ = device;
    const auto &dtype{model_config->get_dtype()};

    model_ = this->register_module<LlamaModel>("model", model_config, device, rank_info, attention_backend);
    lm_head_ = this->register_module<infinicore::nn::Linear>("lm_head", model_config->get<size_t>("hidden_size"), model_config->get<size_t>("vocab_size"), false,
                              dtype, device);
}

LlamaForCausalLM::Output LlamaForCausalLM::forward(const Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();
    auto past_sequence_lengths = input.past_sequence_lengths;
    auto total_sequence_length = input.total_sequence_lengths;
    auto input_offsets = input.input_offsets;
    auto cu_seqlens = input.cu_seqlens;
    auto block_tables = input.block_tables;
    auto slot_mapping = input.slot_mapping;

    auto hidden_states = model_->forward(
        input_ids, position_ids, past_sequence_lengths, total_sequence_length, input_offsets, cu_seqlens, block_tables, slot_mapping);

    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

infinicore::Tensor LlamaForCausalLM::logits_from_hidden(const infinicore::Tensor &hidden_states) const {
    return lm_head_->forward(const_cast<infinicore::Tensor &>(hidden_states));
}

void LlamaForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    cache_config_ = cache_config->unique_copy();
    model_->reset_cache(cache_config_.get());
}

const cache::CacheConfig *LlamaForCausalLM::get_cache_config() const {
    return cache_config_.get();
}

} // namespace infinilm::models::llama_legacy
