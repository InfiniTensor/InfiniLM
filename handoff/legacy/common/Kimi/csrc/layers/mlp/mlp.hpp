#pragma once
#include <typeinfo>

#include "../../config/model_config.hpp"
#include "../linear/linear.hpp"
#include "../../weights/weights.hpp"

namespace infinilm::backends { enum class AttentionBackend; }
namespace infinilm::layers::moe { struct EPConfig; }

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
class MLP : public infinilm::weights::ModelWeights {
public:
    MLP(infinilm::runtime::Runtime &runtime, std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx, const engine::distributed::RankInfo &rank_info,
        backends::AttentionBackend attention_backend, const layers::moe::EPConfig &ep_config);

    /**
     * @brief Forward pass: compute MLP output
     *
     * @param hidden_states Input tensor of shape [batch, seq_len, hidden_size]
     * @return Output tensor of shape [batch, seq_len, hidden_size]
     */
    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                              const infinilm::runtime::Tensor &hidden_states) const;

    // Owner-local enumeration only; no new child/parameter registration.
    void append_w4_layers(std::vector<std::shared_ptr<infinilm::nn::BaseLinear>> &layers,
                          infinilm::quantization::QuantScheme expected_scheme) const {
        if (!gate_up_proj_ || typeid(*gate_up_proj_) != typeid(infinilm::layers::linear::GateUpParallelLinear) ||
            gate_up_proj_->get_quantization()->get_quant_scheme() != expected_scheme)
            throw std::invalid_argument("Qwen3 hidden GateUp must be exact homogeneous W4 owner");
        layers.push_back(gate_up_proj_);
    }

    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override {
        gate_up_proj_->process_weights_after_loading(runtime, pre_transpose);
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
