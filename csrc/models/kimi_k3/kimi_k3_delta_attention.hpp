#pragma once

#include "../../config/model_config.hpp"
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
                    size_t layer_idx,
                    size_t state_channel_offset,
                    const infinicore::DataType &dtype,
                    const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &input) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    size_t layer_idx_{0};
    size_t local_channels_{0};
    size_t state_channel_offset_{0};
};

class KimiK3DeltaAttention : public infinicore::nn::Module {
public:
    KimiK3DeltaAttention(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    size_t layer_idx_{0};
    size_t local_num_heads_{0};
    size_t head_dim_{0};
    size_t local_projection_size_{0};
    float gate_lower_bound_{-5.0f};

    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, q_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, k_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, v_proj);
    INFINICORE_NN_MODULE(KimiK3ShortConv, q_conv1d);
    INFINICORE_NN_MODULE(KimiK3ShortConv, k_conv1d);
    INFINICORE_NN_MODULE(KimiK3ShortConv, v_conv1d);
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
