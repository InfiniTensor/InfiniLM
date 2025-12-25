#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"
#include <optional>

namespace infinilm::models::llama {

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

std::pair<infinicore::Tensor, infinicore::Tensor> LlamaDecoderLayer::forward(
    const infinicore::Tensor &hidden_states,
    const infinicore::Tensor &position_ids,
    std::shared_ptr<infinilm::cache::Cache> kv_cache,
    const infinicore::Tensor &cache_positions,
    const std::optional<infinicore::Tensor> &residual_in) const {
    
    infinicore::Tensor normed_states;
    infinicore::Tensor residual;
    
    // 1. Pre-attention layer normalization with optional residual add from previous layer
    if (residual_in.has_value()) {
        // Fuse previous layer's MLP residual add with current layer's input normalization
        // This avoids a separate add operation: residual_in + hidden_states
        auto [normed_result, add_result] = infinicore::op::add_rms_norm(
            residual_in.value(), hidden_states,
            input_layernorm_->weight(),
            static_cast<float>(input_layernorm_->eps()));
        normed_states = normed_result;
        residual = add_result;  // This is residual_in + hidden_states
    } else {
        // First layer: no residual to add, just normalize
        normed_states = input_layernorm_->forward(hidden_states);
        residual = hidden_states;
    }

    // 2. Self-attention with residual connection
    auto attn_output = self_attn_->forward(normed_states, position_ids, kv_cache, cache_positions);

    // 3. Add attention residual and apply post-attention layer normalization (fused)
    auto [normed_states_result, add_result] = infinicore::op::add_rms_norm(
        residual, attn_output, 
        post_attention_layernorm_->weight(), 
        static_cast<float>(post_attention_layernorm_->eps()));
    
    normed_states = normed_states_result;
    residual = add_result;  // Save for MLP residual connection

    // 4. MLP
    auto mlp_output = mlp_->forward(normed_states);

    // Return (mlp_output, residual) WITHOUT doing the final add
    // Next layer will fuse this add with its input_layernorm using add_rms_norm
    return std::make_pair(mlp_output, residual);
}

} // namespace infinilm::models::llama
