#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <optional>

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
LlamaDecoderLayer::LlamaDecoderLayer(const LlamaConfig &config,
                                     const infinicore::Device &device,
                                     size_t layer_idx,
                                     engine::distributed::RankInfo rank_info) : layer_idx_(layer_idx), rank_info_(rank_info) {
    const auto &dtype{config.dtype};

    // Initialize layer normalization layers
    INFINICORE_NN_MODULE_INIT(input_layernorm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, config.hidden_size, config.rms_norm_eps,
                              dtype, device);

    // Initialize attention and MLP modules
    INFINICORE_NN_MODULE_INIT(self_attn, config, device, layer_idx, rank_info_);
    INFINICORE_NN_MODULE_INIT(mlp, config, device, rank_info_);
}

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

std::tuple<infinicore::Tensor, infinicore::Tensor>
LlamaDecoderLayer::forward(infinicore::Tensor &hidden_states,
                           infinicore::Tensor &residual,
                           const infinicore::Tensor &position_ids,
                           std::shared_ptr<infinilm::cache::Cache> kv_cache,
                           std::optional<infinicore::Tensor> past_sequence_lengths,
                           std::optional<infinicore::Tensor> total_sequence_lengths,
                           std::optional<infinicore::Tensor> input_offsets,
                           std::optional<infinicore::Tensor> block_tables,
                           std::optional<infinicore::Tensor> slot_mapping,
                           std::optional<int64_t> max_sequence_length) const {
    // 1. Attention layer normalization
    input_layernorm_->forward_inplace(hidden_states, residual);

    // 2. Self-attention
    hidden_states = self_attn_->forward(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping, max_sequence_length);

    // 3. Post-attention layer normalization
    post_attention_layernorm_->forward_inplace(hidden_states, residual);

    // 4. MLP
    hidden_states = mlp_->forward(hidden_states);

    return std::make_tuple(hidden_states, residual);
}

// LlamaDecoderLayer::forward(infinicore::Tensor &hidden_states,
//                            infinicore::Tensor &res,
//                            const infinicore::Tensor &position_ids,
//                            std::shared_ptr<infinilm::cache::Cache> kv_cache,
//                            std::optional<infinicore::Tensor> past_sequence_lengths,
//                            std::optional<infinicore::Tensor> total_sequence_lengths,
//                            std::optional<infinicore::Tensor> input_offsets,
//                            std::optional<infinicore::Tensor> block_tables,
//                            std::optional<infinicore::Tensor> slot_mapping) const {
//     // Save residual for attention
//     infinicore::Tensor residual = hidden_states;

//     // 1. Pre-attention layer normalization
//     hidden_states = input_layernorm_->forward(hidden_states);

//     // 2. Self-attention with residual connection
//     auto attn_output = self_attn_->forward(hidden_states, position_ids, kv_cache, past_sequence_lengths, total_sequence_lengths, input_offsets, block_tables, slot_mapping);

//     // Add residual: hidden_states = hidden_states + attn_output
//     hidden_states = infinicore::op::add(residual, attn_output);
//     // Save residual for MLP
//     residual = hidden_states;

//     // 3. Post-attention layer normalization
//     hidden_states = post_attention_layernorm_->forward(hidden_states);

//     // 4. MLP with residual connection
//     auto mlp_output = mlp_->forward(hidden_states);

//     // Add residual: output = output + mlp_output
//     auto output = infinicore::op::add(residual, mlp_output);

//     return std::make_tuple(output, res);
// }

} // namespace infinilm::models::llama
