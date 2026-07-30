#pragma once

#include "../../layers/common_modules.hpp"

namespace infinilm::models::qwen3_next {
class Qwen3NextCausalConv1D : public infinicore::nn::Module {
public:
    Qwen3NextCausalConv1D(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &qkv) const;
    void process_weights_after_loading() override;

private:
    infinicore::nn::Parameter weight_;
    // Contiguous vendor layout [kernel, local_channels], materialized once
    // after checkpoint loading and TP slicing.
    infinicore::Tensor vendor_weight_;
    // Scratch [pool, kernel - 1, local_channels] used only by vendor prefill.
    // Decode keeps the persistent cache contiguous as [pool, channels, kernel - 1].
    mutable infinicore::Tensor vendor_state_;

    size_t layer_idx_;
    size_t local_conv_dim_;
    size_t full_qk_dim_;
    size_t full_v_dim_;
    size_t local_qk_dim_;
    size_t local_v_dim_;
    size_t conv_kernel_dim_;
    size_t tp_rank_;
    size_t tp_size_;
};

class Qwen3NextGatedDeltaNet : public infinicore::nn::Module {
public:
    Qwen3NextGatedDeltaNet(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;
    std::tuple<infinicore::Tensor, infinicore::Tensor>
    forward_add_rmsnorm(
        const infinicore::Tensor &hidden_states,
        const infinicore::Tensor &residual,
        const infinicore::Tensor &gamma,
        float epsilon) const;

private:
    infinicore::Tensor forward_projected_(
        const infinicore::Tensor &hidden_states) const;

    std::shared_ptr<layers::linear::QKVParallelLinear> in_proj_qkv_;
    std::shared_ptr<layers::linear::QKVParallelLinear> in_proj_zab_;
    std::shared_ptr<Qwen3NextCausalConv1D> conv1d_;

    INFINICORE_NN_PARAMETER(dt_bias);
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);
    std::shared_ptr<layers::linear::RowParallelLinear> out_proj_;

    size_t layer_idx_;
    size_t local_num_value_heads_;
    size_t local_num_key_heads_;
    size_t key_head_dim_;
    size_t value_head_dim_;
    size_t local_key_dim_;
    size_t local_value_dim_;
};

} // namespace infinilm::models::qwen3_next
