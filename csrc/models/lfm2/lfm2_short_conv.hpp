#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/parameter.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::lfm2 {

/**
 * LFM2's gated depthwise causal convolution block.
 *
 * The Hugging Face reference computes
 *   (B, C, x) = split(in_proj(hidden_states))
 *   y = out_proj(C * depthwise_causal_conv1d(B * x))
 * and keeps the last K - 1 inputs as the recurrent convolution state.
 */
class Lfm2ShortConv : public infinicore::nn::Module {
public:
    Lfm2ShortConv(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

    void reset_runtime_state() const override {
        in_proj_->reset_runtime_state();
        out_proj_->reset_runtime_state();
    }

private:
    infinicore::Tensor causal_depthwise_conv_(const infinicore::Tensor &input) const;

    size_t layer_idx_{0};
    size_t hidden_size_{0};
    size_t kernel_size_{0};
    bool use_bias_{false};

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, in_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, out_proj);
    INFINICORE_NN_PARAMETER(conv_weight);
    INFINICORE_NN_PARAMETER(conv_bias);
};

} // namespace infinilm::models::lfm2
