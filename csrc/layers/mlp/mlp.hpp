#pragma once

#include "../../config/model_config.hpp"
#include "infinicore/device.hpp"
#include "../linear/fused_linear.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include "../../engine/distributed/distributed.hpp"

namespace infinilm::layers::mlp {

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
class MLP : public infinicore::nn::Module {
public:
    /**
     * @brief Construct MLP module
     *
     * @param config Model configuration
     * @param device Device to create tensors on
     * @param dtype Optional data type for model parameters (defaults to F32)
     */
    MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

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
    INFINICORE_NN_MODULE(infinilm::layers::linear::GateUpParallelLinear, gate_up_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RowParallelLinear, down_proj);

    std::shared_ptr<infinilm::config::ModelConfig> model_config_;

    size_t hidden_size_;
    size_t intermediate_size_;
    bool use_bias_;
};

} // namespace infinilm::layers::mlp
