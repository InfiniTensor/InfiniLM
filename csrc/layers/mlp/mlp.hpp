#pragma once

#include "../../config/model_config.hpp"
#include "../linear/linear.hpp"
#include "infinicore/nn/module.hpp"

namespace infinilm::layers::mlp {

/**
 * @brief MLP (Feed-Forward Network) module.
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
     * @param model_config: Model configuration.
     * @param device Device to create tensors on
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

    void process_weights_after_loading() override {
        gate_up_proj_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        gate_up_proj_->reset_runtime_state();
    }

    // Module information
    size_t hidden_size() const { return hidden_size_; }
    size_t intermediate_size() const { return intermediate_size_; }

protected:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;

    size_t hidden_size_;
    size_t intermediate_size_;
    bool use_bias_;
};

} // namespace infinilm::layers::mlp
