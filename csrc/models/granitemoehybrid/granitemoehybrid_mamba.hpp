#pragma once

#include "../../layers/common_modules.hpp"

#include <memory>
#include <optional>

namespace infinilm::models::granitemoehybrid {

class GraniteMoeHybridCausalConv1d : public infinicore::nn::Module {
public:
    GraniteMoeHybridCausalConv1d(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        size_t layer_idx,
        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input) const;
    void process_weights_after_loading() override;

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);
    size_t layer_idx_;
    bool use_bias_;
    size_t tp_size_;
    size_t tp_rank_;
    size_t conv_kernel_dim_;
    size_t full_x_dim_;
    size_t full_bc_dim_;
    size_t local_x_dim_;
    size_t local_bc_dim_;
    size_t local_conv_dim_;
    size_t bc_replicas_;
};

class GraniteMoeHybridRMSNormGated : public infinicore::nn::Module {
public:
    GraniteMoeHybridRMSNormGated(
        std::shared_ptr<infinilm::config::ModelConfig> model_config,
        const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               std::optional<infinicore::Tensor> gate = std::nullopt) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    double eps_;
};

class GraniteMoeHybridMamba : public infinicore::nn::Module {
public:
    GraniteMoeHybridMamba(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          size_t layer_idx,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;
    void process_weights_after_loading() override {
        in_proj_->process_weights_after_loading();
    }

    void reset_runtime_state() const override {
        in_proj_->reset_runtime_state();
        ssm_state_ = infinicore::Tensor();
    }

private:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, in_proj);
    INFINICORE_NN_MODULE(GraniteMoeHybridCausalConv1d, conv1d);
    INFINICORE_NN_PARAMETER(dt_bias);
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_MODULE(GraniteMoeHybridRMSNormGated, norm);
    INFINICORE_NN_PARAMETER(D);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, out_proj);
    INFINICORE_NN_PARAMETER(out_proj_bias);

    size_t intermediate_size_;
    size_t num_heads_;
    size_t num_groups_;
    size_t head_dim_;
    size_t state_size_;
    size_t conv_dim_;

    mutable infinicore::Tensor ssm_state_;
};

} // namespace infinilm::models::granitemoehybrid
