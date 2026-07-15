#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/router/topk_router.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <infiniccl.h>

#include <memory>
#include <optional>
#include <string>
#include <tuple>

namespace infinilm::models::deepseek_v2 {

using DeepseekV2MLP = infinilm::layers::mlp::MLP;

using DeepseekV2TopKRouter = infinilm::layers::moe::TopKRouter;

class DeepseekV2Experts final : public infinicore::nn::Module {
public:
    DeepseekV2Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &top_k_index,
                               const infinicore::Tensor &top_k_weights,
                               std::optional<infinicore::Tensor> shared_output = std::nullopt) const;

private:
    infinicore::Tensor w1_;
    infinicore::Tensor w2_;
    size_t hidden_size_{0};
    size_t local_moe_intermediate_size_{0};
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    size_t tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};

class DeepseekV2MoE final : public infinicore::nn::Module {
public:
    DeepseekV2MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

private:
    INFINICORE_NN_MODULE(DeepseekV2TopKRouter, gate);
    INFINICORE_NN_MODULE(DeepseekV2Experts, experts);
    INFINICORE_NN_MODULE(DeepseekV2MLP, shared_experts);
    bool has_shared_experts_{false};
};

} // namespace infinilm::models::deepseek_v2
