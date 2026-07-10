#pragma once

#include "../../layers/common_modules.hpp"
#include "../../layers/linear/linear.hpp"
#include "infinicore/nn/parameter.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>
#include <vector>

namespace infinilm::models::ernie4_5_vl {

class Ernie45TopKRouter : public infinicore::nn::Module {
public:
    Ernie45TopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(const infinicore::Tensor &hidden_states, const infinicore::Tensor &correction_bias = infinicore::Tensor()) const;

protected:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(weight_1);
    size_t num_experts_per_tok_{0};
    bool norm_topk_prob_{false};
};

class Ernie45ExpertMLP : public infinicore::nn::Module {
public:
    Ernie45ExpertMLP(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     size_t intermediate_size,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;
    void set_alpha(float alpha) { down_proj_->set_alpha(alpha); }

protected:
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, gate_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ColumnParallelLinear, up_proj);
    INFINICORE_NN_MODULE(infinilm::layers::linear::RowParallelLinear, down_proj);
};

class Ernie45Experts : public infinicore::nn::Module {
public:
    Ernie45Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                   const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &top_k_index,
                               const infinicore::Tensor &top_k_weights) const;

protected:
    INFINICORE_NN_PARAMETER(w1);
    INFINICORE_NN_PARAMETER(w2);
    INFINICORE_NN_PARAMETER(b1);
    INFINICORE_NN_PARAMETER(b2);
    size_t num_experts_per_tok_{0};
    size_t num_text_experts_{0};
    bool use_bias_{false};
};

class Ernie45MoE : public infinicore::nn::Module {
public:
    Ernie45MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
               const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(Ernie45TopKRouter, gate);
    INFINICORE_NN_MODULE(Ernie45Experts, experts);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, shared_experts);
    infinicore::nn::Parameter e_score_correction_bias_;
    size_t num_experts_{0};
    bool has_shared_experts_{false};
};

} // namespace infinilm::models::ernie4_5_vl
