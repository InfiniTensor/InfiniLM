#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/mlp/mlp.hpp"
#include "../../layers/moe/common/moe_types.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include <infinicore/nn/module.hpp>
#include <infinicore/tensor.hpp>

#include <cstddef>
#include <memory>
#include <vector>

namespace infinilm::models::kimi_k25 {

class KimiK25MXFP4Experts : public infinicore::nn::Module {
public:
    KimiK25MXFP4Experts(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                        const infinicore::Device &device);

    const infinilm::layers::moe::MoeWeights &moe_weights() const;
    void process_weights_after_loading() override;

protected:
    std::vector<infinicore::nn::Parameter> packed_parameters_;
    infinilm::layers::moe::MoeWeights moe_weights_;
    size_t num_experts_{0};
    size_t hidden_size_{0};
    size_t local_intermediate_size_{0};
    infinicore::DataType dtype_;
    infinicore::Device device_;
};

class KimiK25MoE : public infinicore::nn::Module {
public:
    KimiK25MoE(std::shared_ptr<infinilm::config::ModelConfig> model_config,
               size_t layer_idx,
               const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    INFINICORE_NN_MODULE(infinilm::layers::moe::TopKRouter, gate);
    INFINICORE_NN_MODULE(infinilm::layers::moe::FusedMoeExperts, experts);
    INFINICORE_NN_MODULE(KimiK25MXFP4Experts, mxfp4_experts);
    INFINICORE_NN_MODULE(infinilm::layers::moe::FusedMoE, fused_moe);
    INFINICORE_NN_MODULE(infinilm::layers::mlp::MLP, shared_experts);
    bool uses_mxfp4_experts_{false};
    float routed_scaling_factor_{1.0f};
};

std::shared_ptr<infinilm::config::ModelConfig>
make_kimi_mlp_config(const std::shared_ptr<infinilm::config::ModelConfig> &model_config,
                     size_t intermediate_size);

} // namespace infinilm::models::kimi_k25
