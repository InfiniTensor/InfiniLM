#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/common_modules.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "infinicore/device.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <infiniccl.h>

#include <memory>
#include <tuple>
#include <vector>

namespace infinilm::models::deepseek_v2 {

using DeepseekV2MLP = infinilm::layers::mlp::MLP;
using DeepseekV2ExpertMLP = infinilm::layers::MoeMLP;

class DeepseekV2TopKRouter : public infinicore::nn::Module {
public:
    DeepseekV2TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_PARAMETER(weight);
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    bool norm_topk_prob_{false};
};

class DeepseekV2Experts : public infinicore::nn::Module {
public:
    DeepseekV2Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &top_k_index,
                               const infinicore::Tensor &top_k_weights) const;

protected:
    infinicore::Tensor forward_cpu_routed_(const infinicore::Tensor &hidden_states,
                                           const infinicore::Tensor &top_k_index,
                                           const infinicore::Tensor &top_k_weights) const;

    INFINICORE_NN_MODULE_VEC(DeepseekV2ExpertMLP, experts);
    std::vector<infinicore::Tensor> gate_weights_;
    std::vector<infinicore::Tensor> up_weights_;
    std::vector<infinicore::Tensor> down_weights_;
    size_t hidden_size_{0};
    size_t moe_intermediate_size_{0};
    size_t local_moe_intermediate_size_{0};
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    size_t tp_size_{1};
    infinicclComm_t communicator_{nullptr};
};

class DeepseekV2MoE : public infinicore::nn::Module {
public:
    DeepseekV2MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(DeepseekV2TopKRouter, gate);
    INFINICORE_NN_MODULE(DeepseekV2Experts, experts);
    INFINICORE_NN_MODULE(DeepseekV2MLP, shared_experts);
    bool has_shared_experts_{false};
};

} // namespace infinilm::models::deepseek_v2
