#pragma once

#include "../../layers/fused_linear.hpp"
#include "llama_config.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include "llama_config.hpp"

#include "../../engine/distributed/distributed.hpp"

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
     * @param dtype Optional data type for model parameters (defaults to F32)
     */
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
    LlamaMLP(const LlamaConfig &config,
             const infinicore::Device &device,
             engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    LlamaMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
             const infinicore::Device &device,
             engine::distributed::RankInfo rank_info = engine::distributed::RankInfo());

    /**
     * @brief Forward pass: compute MLP output
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    // Module information
    size_t hidden_size() const { return hidden_size_; }
    size_t intermediate_size() const { return intermediate_size_; }

protected:
    INFINICORE_NN_MODULE(layers::GateUpParallelLinear, gate_up_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RowParallelLinear, down_proj);

    engine::distributed::RankInfo rank_info_;
    size_t hidden_size_;
    size_t intermediate_size_;
    bool use_bias_;

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

} // namespace infinilm::models::llama
