#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <optional>

namespace infinilm::models::llama {

LlamaDecoderLayer::LlamaDecoderLayer(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                                     const infinicore::Device &device,
                                     size_t layer_idx,
                                     engine::distributed::RankInfo rank_info) : model_config_(model_config), layer_idx_(layer_idx), rank_info_(rank_info) {
    const auto &dtype{model_config_->get_dtype()};
    // Initialize layer normalization layers
    INFINICORE_NN_MODULE_INIT(input_layernorm, model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, model_config_->get<size_t>("hidden_size"), model_config_->get<double>("rms_norm_eps"),
                              dtype, device);

    // Initialize attention and MLP modules
    INFINICORE_NN_MODULE_INIT(self_attn, model_config_, device, layer_idx, rank_info_);
    INFINICORE_NN_MODULE_INIT(mlp, model_config_, device, rank_info_);
}

infinicore::Tensor LlamaDecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &position_ids,
                                              std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                              std::optional<infinicore::Tensor> past_sequence_lengths,
                                              std::optional<infinicore::Tensor> total_sequence_lengths,
                                              std::optional<infinicore::Tensor> input_offsets,
                                              std::optional<infinicore::Tensor> block_tables,
                                              std::optional<infinicore::Tensor> slot_mapping) const {
    // Save residual for attention
    auto residual = hidden_states;

    // 1. Pre-attention layer normalization
    auto normed_states = input_layernorm_->forward(hidden_states);

    // 2. Self-attention with residual connection
    auto attn_output = self_attn_->forward(normed_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping);

    // Add residual: hidden_states = hidden_states + attn_output
    auto output = infinicore::op::add(residual, attn_output);
    // Save residual for MLP
    residual = output;

    // 3. Post-attention layer normalization
    normed_states = post_attention_layernorm_->forward(output);

    // 4. MLP with residual connection
    auto mlp_output = mlp_->forward(normed_states);

    // Add residual: output = output + mlp_output
    output = infinicore::op::add(residual, mlp_output);

    return output;
}

} // namespace infinilm::models::llama
