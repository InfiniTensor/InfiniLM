#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <optional>

namespace infinilm::models::llama_legacy {

LlamaDecoderLayer::LlamaDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device,
                                     size_t layer_idx,
                                     engine::distributed::RankInfo rank_info,
                                     backends::AttentionBackend attention_backend) : model_config_(model_config), layer_idx_(layer_idx), rank_info_(rank_info) {
    const auto &dtype{model_config_->get_dtype()};
    input_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("input_layernorm", model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                              dtype, device);
    post_attention_layernorm_ = this->register_module<infinicore::nn::RMSNorm>("post_attention_layernorm", model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                              dtype, device);

    self_attn_ = this->register_module<LlamaAttention>("self_attn", model_config_, device, layer_idx, rank_info_, attention_backend);
    mlp_ = this->register_module<LlamaMLP>("mlp", model_config_, device, rank_info_);
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
LlamaDecoderLayer::forward(infinicore::Tensor &hidden_states,
                           infinicore::Tensor &residual,
                           const infinicore::Tensor &position_ids,
                           std::shared_ptr<infinilm::cache::Cache> kv_cache,
                           std::optional<infinicore::Tensor> past_sequence_lengths,
                           std::optional<infinicore::Tensor> total_sequence_lengths,
                           std::optional<infinicore::Tensor> input_offsets,
                           std::optional<infinicore::Tensor> cu_seqlens,
                           std::optional<infinicore::Tensor> block_tables,
                           std::optional<infinicore::Tensor> slot_mapping) const {
    // 1. Attention layer normalization
    input_layernorm_->forward_inplace(hidden_states, residual);

    // 2. Self-attention
    hidden_states = self_attn_->forward(
        hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, cu_seqlens, block_tables, slot_mapping);

    // 3. Post-attention layer normalization
    post_attention_layernorm_->forward_inplace(hidden_states, residual);

    // 4. MLP
    hidden_states = mlp_->forward(hidden_states);

    return std::make_tuple(hidden_states, residual);
}

} // namespace infinilm::models::llama_legacy
