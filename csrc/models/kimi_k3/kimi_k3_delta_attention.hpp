#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/fused_linear.hpp"
#include "../../layers/linear/linear.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::kimi_k3 {

class KimiK3ShortConv : public infinicore::nn::Module {
public:
    KimiK3ShortConv(size_t full_channels,
                    size_t kernel_size,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);

    infinicore::Tensor weight() const { return weight_; }

private:
    INFINICORE_NN_PARAMETER(weight);
};

class KimiK3DeltaAttention : public infinicore::nn::Module {
public:
    KimiK3DeltaAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

private:
    size_t layer_idx_{0};
    size_t local_num_heads_{0};
    size_t head_dim_{0};
    size_t local_projection_size_{0};
    size_t conv_kernel_size_{0};
    float gate_lower_bound_{-5.0f};

    std::shared_ptr<infinilm::layers::linear::QKVParallelLinear> qkv_proj_;
    INFINICORE_NN_MODULE(KimiK3ShortConv, q_conv1d);
    INFINICORE_NN_MODULE(KimiK3ShortConv, k_conv1d);
    INFINICORE_NN_MODULE(KimiK3ShortConv, v_conv1d);
    infinicore::Tensor packed_conv_weight_;
    INFINICORE_NN_PARAMETER(A_log);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, f_a_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, f_b_proj);
    INFINICORE_NN_PARAMETER(dt_bias);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, b_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, g_proj);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, o_norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, o_proj);
};

} // namespace infinilm::models::kimi_k3
