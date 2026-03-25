#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3_next {
using Qwen3Next_Fake_RMSNormGated = infinicore::nn::RMSNorm;

class FakeConv1d : public infinicore::nn::Module {
public:
    FakeConv1d(size_t in_channels,
               size_t out_channels,
               size_t kernel_size,
               size_t stride,
               size_t padding,
               size_t dilation,
               size_t groups,
               bool bias,
               const infinicore::DataType dtype,
               const infinicore::Device device);

private:
    size_t layer_idx_;
    INFINICORE_NN_PARAMETER(weight);
};

class Qwen3NextGatedDeltaNet : public infinicore::nn::Module {
public:
    Qwen3NextGatedDeltaNet(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t layer_idx_;

    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, in_proj_qkvz);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, in_proj_ba);
    INFINICORE_NN_MODULE(FakeConv1d, conv1d);
    INFINICORE_NN_PARAMETER(dt_bias);
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_MODULE(Qwen3Next_Fake_RMSNormGated, norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, out_proj);
};

} // namespace infinilm::models::qwen3_next
