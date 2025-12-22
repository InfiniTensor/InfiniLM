#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"

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

infinicore::Tensor LlamaDecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                              const infinicore::Tensor &position_ids,
                                              std::shared_ptr<infinilm::cache::Cache> kv_cache,
                                              const infinicore::Tensor &cache_positions) const {
    // Save residual for attention
    auto residual = hidden_states;

    // 1. Pre-attention layer normalization
    auto normed_states = input_layernorm_->forward(hidden_states);

    // 2. Self-attention with residual connection
    auto attn_output = self_attn_->forward(normed_states, position_ids, kv_cache, cache_positions);

    // Add residual and apply post-attention layer normalization (fused)
    // Use fused add_rms_norm operator to reduce kernel launch overhead
    // Fused operator: compute normalized result
    normed_states = infinicore::op::add_rms_norm(
        residual, attn_output, 
        post_attention_layernorm_->weight(), 
        static_cast<float>(post_attention_layernorm_->eps()));
    
    // IMPORTANT: Save residual for MLP must be the ADD result (before normalization),
    // not the normalized result. This matches the behavior of separate operators.
    auto output = infinicore::op::add(residual, attn_output);
    residual = output;

    // 4. MLP with residual connection
    auto mlp_output = mlp_->forward(normed_states);

    // Add residual: output = residual + mlp_output
    output = infinicore::op::add(residual, mlp_output);

    return output;
}

} // namespace infinilm::models::llama
