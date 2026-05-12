#pragma once

#include "../../layers/linear/fused_linear.hpp"
#include "llama_config.hpp"

#include "../../config/model_config.hpp"
#include "infinicore/device.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include "llama_config.hpp"

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::models::llama_legacy {

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
    std::shared_ptr<layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::nn::RowParallelLinear> down_proj_;

    engine::distributed::RankInfo rank_info_;
    size_t hidden_size_;
    size_t intermediate_size_;
    bool use_bias_;

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;
};

} // namespace infinilm::models::llama_legacy
