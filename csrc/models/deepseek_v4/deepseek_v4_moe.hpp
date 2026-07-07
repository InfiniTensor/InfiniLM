#pragma once

#include "../../config/model_config.hpp"
#include "deepseek_v4_mlp.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"
#include <infiniccl.h>

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace infinilm::models::deepseek_v4 {

class DeepseekV4TopKRouter : public infinicore::nn::Module {
public:
    DeepseekV4TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states,
                                                               const infinicore::Tensor &input_ids = infinicore::Tensor()) const;
    infinicore::Tensor weight() const { return weight_; }
    infinicore::Tensor bias() const { return bias_; }
    bool has_bias() const { return !bias_.empty(); }

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(bias);

    size_t hidden_size_{0};
    size_t num_experts_{0};
    size_t num_experts_per_tok_{0};
    float routed_scaling_{1.0f};
    std::string scoring_func_;
    bool norm_topk_prob_{true};
    size_t layer_idx_{0};
};

class DeepseekV4HashRouter : public infinicore::nn::Module {
public:
    DeepseekV4HashRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         size_t layer_idx,
                         const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states,
                                                               const infinicore::Tensor &input_ids = infinicore::Tensor()) const;
    infinicore::Tensor weight() const { return weight_; }
    infinicore::Tensor tid2eid() const { return tid2eid_; }
    bool has_tid2eid() const { return !tid2eid_.empty(); }

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(tid2eid);

    size_t hidden_size_{0};
    size_t num_experts_{0};
    size_t num_experts_per_tok_{0};
    float routed_scaling_{1.0f};
    std::string scoring_func_;
    bool norm_topk_prob_{true};
    size_t layer_idx_{0};
};

class DeepseekV4Experts : public infinicore::nn::Module {
public:
    DeepseekV4Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &top_k_index,
                               const infinicore::Tensor &top_k_weights) const;

private:
    infinicore::Tensor forward_cpu_routed_(const infinicore::Tensor &hidden_states,
                                           const infinicore::Tensor &top_k_index,
                                           const infinicore::Tensor &top_k_weights) const;
    INFINICORE_NN_MODULE_VEC(DeepseekV4MLP, experts);
    std::vector<infinicore::Tensor> gate_weights_;
    std::vector<infinicore::Tensor> up_weights_;
    std::vector<infinicore::Tensor> down_weights_;
    std::vector<infinicore::Tensor> gate_weight_scales_;
    std::vector<infinicore::Tensor> up_weight_scales_;
    std::vector<infinicore::Tensor> down_weight_scales_;

    size_t hidden_size_{0};
    size_t moe_intermediate_size_{0};
    size_t local_moe_intermediate_size_{0};
    size_t num_experts_{0};
    size_t num_experts_per_tok_{0};
    size_t tp_size_{1};
    bool use_fused_moe_{false};
    infinicclComm_t communicator_{nullptr};
};

class DeepseekV4MoE : public infinicore::nn::Module {
public:
    DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  const infinicore::Device &device);
    DeepseekV4MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                  size_t layer_idx,
                  const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &input_ids = infinicore::Tensor()) const;

private:
    using DeepseekV4Gate = std::variant<std::shared_ptr<DeepseekV4HashRouter>, std::shared_ptr<DeepseekV4TopKRouter>>;
    DeepseekV4Gate gate_;

    INFINICORE_NN_MODULE(DeepseekV4Experts, experts);
    INFINICORE_NN_MODULE(DeepseekV4MLP, shared_experts);

    size_t hidden_size_{0};
    size_t layer_idx_{0};
    size_t tp_size_{1};
    infinicclComm_t communicator_{nullptr};
    bool has_shared_experts_{false};

    char *f32_allreduce_env;
};

} // namespace infinilm::models::deepseek_v4
