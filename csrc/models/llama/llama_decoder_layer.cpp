#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"

#include <optional>

namespace infinilm::models::llama {

LlamaDecoderLayer::LlamaDecoderLayer(std::shared_ptr<infinilm::config::global_config::GlobalConfig> global_config,
                                     const infinicore::Device &device,
                                     size_t layer_idx,
                                     engine::distributed::RankInfo rank_info) : global_config_(global_config), layer_idx_(layer_idx), rank_info_(rank_info) {
    const auto &dtype{global_config_->get_dtype()};

    // Initialize layer normalization layers
    INFINICORE_NN_MODULE_INIT(input_layernorm, global_config_->get<size_t>("hidden_size"), global_config_->get<double>("rms_norm_eps"),
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, global_config_->get<size_t>("hidden_size"), global_config_->get<double>("rms_norm_eps"),
                              dtype, device);

    // Initialize attention and MLP modules
    INFINICORE_NN_MODULE_INIT(self_attn, global_config, device, layer_idx, rank_info_);
    INFINICORE_NN_MODULE_INIT(mlp, global_config, device, rank_info_);
}

std::tuple<infinicore::Tensor, infinicore::Tensor>
LlamaDecoderLayer::forward(infinicore::Tensor &hidden_states,
                           infinicore::Tensor &residual,
                           const infinicore::Tensor &position_ids,
                           std::shared_ptr<infinilm::cache::Cache> kv_cache,
                           std::optional<infinicore::Tensor> past_sequence_lengths,
                           std::optional<infinicore::Tensor> total_sequence_lengths,
                           std::optional<infinicore::Tensor> input_offsets,
                           std::optional<infinicore::Tensor> block_tables,
                           std::optional<infinicore::Tensor> slot_mapping) const {
    // 1. Attention layer normalization
    input_layernorm_->forward_inplace(hidden_states, residual);

    // 2. Self-attention
    hidden_states = self_attn_->forward(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping);

    // 3. Post-attention layer normalization
    post_attention_layernorm_->forward_inplace(hidden_states, residual);

    // 4. MLP
    hidden_states = mlp_->forward(hidden_states);

    return std::make_tuple(hidden_states, residual);
}

} // namespace infinilm::models::llama
