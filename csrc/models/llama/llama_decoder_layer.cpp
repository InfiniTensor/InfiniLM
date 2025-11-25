#include "llama_decoder_layer.hpp"
#include "infinicore/nn/rmsnorm.hpp"
#include "infinicore/ops.hpp"

namespace infinilm::models::llama {

LlamaDecoderLayer::LlamaDecoderLayer(const LlamaConfig &config, const infinicore::Device &device) {
    // Initialize layer normalization layers
    INFINICORE_NN_MODULE_INIT(input_layernorm, config.hidden_size, config.rms_norm_eps,
                              infinicore::DataType::F32, device);
    INFINICORE_NN_MODULE_INIT(post_attention_layernorm, config.hidden_size, config.rms_norm_eps,
                              infinicore::DataType::F32, device);

    // Initialize attention and MLP modules
    INFINICORE_NN_MODULE_INIT(self_attn, config, device);
    INFINICORE_NN_MODULE_INIT(mlp, config, device);
}

infinicore::Tensor LlamaDecoderLayer::forward(const infinicore::Tensor &hidden_states,
                                               const infinicore::Tensor &position_ids,
                                               void *kv_cache,
                                               const HookRegistry *hook_registry,
                                               const std::string &hook_prefix,
                                               int layer_idx) const {
    // Save residual for attention
    auto residual = hidden_states;

    // 1. Pre-attention layer normalization
    auto normed_states = input_layernorm_->forward(hidden_states);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_input_layernorm", normed_states, layer_idx);
    }

    // 2. Self-attention with residual connection
    std::string attn_prefix = hook_prefix.empty() ? "attention" : hook_prefix + "_attention";
    auto attn_output = self_attn_->forward(normed_states, position_ids, kv_cache, hook_registry, attn_prefix, layer_idx);

    // Add residual: hidden_states = hidden_states + attn_output
    auto output = infinicore::op::add(residual, attn_output);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_after_attention_residual", output, layer_idx);
    }

    // Save residual for MLP
    residual = output;

    // 3. Post-attention layer normalization
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_before_post_attention_layernorm", output, layer_idx);
    }
    normed_states = post_attention_layernorm_->forward(output);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_post_attention_layernorm", normed_states, layer_idx);
    }

    // 4. MLP with residual connection
    std::string mlp_prefix = hook_prefix.empty() ? "mlp" : hook_prefix + "_mlp";
    auto mlp_output = mlp_->forward(normed_states, hook_registry, mlp_prefix, layer_idx);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_mlp", mlp_output, layer_idx);
    }

    // Add residual: output = output + mlp_output
    output = infinicore::op::add(residual, mlp_output);
    if (hook_registry && hook_registry->has_hooks()) {
        hook_registry->call_hook(hook_prefix + "_after_mlp_residual", output, layer_idx);
        hook_registry->call_hook(hook_prefix + "_output", output, layer_idx);
    }

    return output;
}

} // namespace infinilm::models::llama
