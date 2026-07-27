#pragma once

#include "../../layers/moe/legacy/moe_mlp.hpp"
#include "infinicore/nn/module.hpp"
#include "infinicore/tensor.hpp"

#include <memory>
#include <tuple>
#include <vector>

namespace infinilm::models::ernie4_5_moe_vl {

class Ernie4_5MoeTopKRouter : public infinicore::nn::Module {
public:
    Ernie4_5MoeTopKRouter(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    std::tuple<infinicore::Tensor, infinicore::Tensor> forward(
        const infinicore::Tensor &hidden_states,
        const infinicore::Tensor &score_correction_bias,
        const std::vector<size_t> *token_groups = nullptr) const;

private:
    INFINICORE_NN_PARAMETER(weight);
    INFINICORE_NN_PARAMETER(weight_1);
    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    size_t num_expert_groups_{1};
    bool norm_topk_prob_{true};
    bool use_correction_bias_{false};
};

class Ernie4_5MoeStatics : public infinicore::nn::Module {
public:
    Ernie4_5MoeStatics(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                       const infinicore::Device &device);

    infinicore::Tensor e_score_correction_bias() const { return e_score_correction_bias_; }

private:
    INFINICORE_NN_PARAMETER(e_score_correction_bias);
};

class Ernie4_5TextMoeBlock : public infinicore::nn::Module {
public:
    Ernie4_5TextMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                         const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states,
                               const infinicore::Tensor &token_type_ids = infinicore::Tensor()) const;

private:
    std::vector<size_t> token_groups_(const infinicore::Tensor &token_type_ids,
                                      const std::vector<size_t> &hidden_shape) const;

    INFINICORE_NN_MODULE(Ernie4_5MoeTopKRouter, gate);
    INFINICORE_NN_MODULE(Ernie4_5MoeStatics, moe_statics);
    INFINICORE_NN_MODULE_VEC(infinilm::layers::moe::legacy::MoeMLP, experts);
    INFINICORE_NN_MODULE(infinilm::layers::moe::legacy::MoeMLP, shared_experts);

    size_t num_experts_per_tok_{0};
    size_t num_experts_{0};
    size_t num_expert_groups_{1};
};

} // namespace infinilm::models::ernie4_5_moe_vl
