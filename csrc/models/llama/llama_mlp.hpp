#pragma once

#include "llama_config.hpp"
#include "llama_hooks.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/tensor.hpp"
#include "infinicore/device.hpp"

namespace infinilm::models::llama {

/**
 * @brief MLP (Feed-Forward Network) module for Llama
 *
 * Implements the MLP block with:
 * - Gate projection
 * - Up projection
 * - Down projection
 * - SiLU activation function
 *
 * Formula: down_proj(SiLU(gate_proj(x)) * up_proj(x))
 */
class LlamaMLP : public infinicore::nn::Module {
public:
    /**
     * @brief Construct LlamaMLP module
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     */
    LlamaMLP(const LlamaConfig &config, const infinicore::Device &device);

    /**
     * @brief Forward pass: compute MLP output
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @param hook_registry Optional hook registry for capturing intermediate values
     * @param hook_prefix Prefix for hook names (e.g., "layer0_mlp")
     * @param layer_idx Layer index for hooks
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     *
     * Note: This is a placeholder forward method. The actual implementation
     * will be added when integrating with the inference engine.
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                                const HookRegistry *hook_registry = nullptr,
                                const std::string &hook_prefix = "",
                                int layer_idx = -1) const;

    // Module information
    size_t hidden_size() const { return hidden_size_; }
    size_t intermediate_size() const { return intermediate_size_; }

protected:
    // Projection layers
    INFINICORE_NN_MODULE(infinicore::nn::Linear, gate_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, up_proj);
    INFINICORE_NN_MODULE(infinicore::nn::Linear, down_proj);

private:
    size_t hidden_size_;
    size_t intermediate_size_;
    bool use_bias_;
};

} // namespace infinilm::models::llama
