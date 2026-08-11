#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include <infiniccl.h>
#include <infinicore/nn/module.hpp>
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <memory>

namespace infinilm::models::kimi_k3 {

struct KimiK3Mxfp4MoeWeights {
    infinicore::Tensor packed_w13;
    infinicore::Tensor w13_scale;
    infinicore::Tensor packed_w2;
    infinicore::Tensor w2_scale;
};

class KimiK3MLP : public infinicore::nn::Module {
public:
    KimiK3MLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              size_t intermediate_size,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, down_proj);
    float situ_beta_{4.0f};
    float situ_linear_beta_{25.0f};
};

class KimiK3Experts : public infinicore::nn::Module {
public:
    KimiK3Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    const KimiK3Mxfp4MoeWeights &mxfp4_weights() const;

private:
    void register_mxfp4_experts();

    KimiK3Mxfp4MoeWeights mxfp4_weights_;
    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t local_intermediate_size_{0};
    size_t tp_rank_{0};
    size_t tp_size_{1};
    infinicore::Device device_;
};

class KimiK3MoE : public infinicore::nn::Module {
public:
    KimiK3MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
              size_t layer_idx,
              const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(infinilm::layers::moe::TopKRouter, gate);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, routed_expert_down_proj);
    INFINICORE_NN_MODULE(KimiK3Experts, experts);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, routed_expert_norm);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, routed_expert_up_proj);
    INFINICORE_NN_MODULE(KimiK3MLP, shared_experts);
    int tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};

std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_k3_subconfig(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                       size_t hidden_size,
                       size_t intermediate_size);

} // namespace infinilm::models::kimi_k3
