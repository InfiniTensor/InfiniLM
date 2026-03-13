#include "minicpm_sala_for_causal_lm.hpp"

#include "infinicore/ops.hpp"
#include <cmath>
#include <stdexcept>

namespace infinilm::models::minicpm_sala {

MiniCPMSALAForCausalLM::MiniCPMSALAForCausalLM(
    std::shared_ptr<infinilm::config::ModelConfig> model_config,
    const infinicore::Device &device,
    engine::distributed::RankInfo rank_info,
    backends::AttentionBackend attention_backend) {
    device_ = device;

    // Match parameter dtype with checkpoint `torch_dtype` (e.g. BF16 for MiniCPM-SALA).
    const auto dtype = model_config->get_dtype();
    INFINICORE_NN_MODULE_INIT(model, model_config, device, rank_info, attention_backend);

    const size_t hidden_size = model_config->get<size_t>("hidden_size");
    const size_t vocab_size = model_config->get<size_t>("vocab_size");

    INFINICORE_NN_MODULE_INIT(lm_head, hidden_size, vocab_size, false, dtype, device);
}

MiniCPMSALAForCausalLM::Output MiniCPMSALAForCausalLM::forward(
    const Input &input) const {
    auto input_ids = input.input_ids.value();
    auto position_ids = input.position_ids.value();

    auto past_sequence_lengths = input.past_sequence_lengths;
    auto total_sequence_lengths = input.total_sequence_lengths;
    auto input_offsets = input.input_offsets;
    auto cu_seqlens = input.cu_seqlens;
    auto block_tables = input.block_tables;
    auto slot_mapping = input.slot_mapping;

    auto hidden_states = model_->forward(
        input_ids,
        position_ids,
        past_sequence_lengths,
        total_sequence_lengths,
        input_offsets,
        cu_seqlens,
        block_tables,
        slot_mapping);

    // MuP lm_head scale baked into lm_head.weight at load time; no forward scaling here.
    auto logits = lm_head_->forward(hidden_states);
    return {logits};
}

void MiniCPMSALAForCausalLM::reset_cache(const cache::CacheConfig *cache_config) {
    cache_config_ = cache_config->unique_copy();
    model_->reset_cache(cache_config_.get());
}

const cache::CacheConfig *MiniCPMSALAForCausalLM::get_cache_config() const {
    return cache_config_.get();
}

} // namespace infinilm::models::minicpm_sala

