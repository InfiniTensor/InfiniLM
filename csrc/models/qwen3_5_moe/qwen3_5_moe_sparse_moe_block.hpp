#pragma once

#include "../../layers/common_modules.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"
#include "../qwen3_moe/qwen3_moe_experts.hpp"
#include "../qwen3_moe/qwen3_moe_topk_router.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::qwen3_5_moe {

class Qwen35MoeSparseMoeBlock final : public infinicore::nn::Module {
public:
    Qwen35MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::models::qwen3_moe::Qwen3MoeTopKRouter> legacy_gate_;
    std::shared_ptr<infinilm::models::qwen3_moe::Qwen3MoeExperts> legacy_experts_;
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::moe::FusedMoeExperts> experts_;
    std::shared_ptr<infinilm::layers::moe::FusedMoE> fused_moe_;
    INFINICORE_NN_MODULE(infinilm::layers::MLP, shared_expert);
    INFINICORE_NN_MODULE(infinilm::layers::linear::ReplicatedLinear, shared_expert_gate);
    bool use_legacy_moe_{false};
};

} // namespace infinilm::models::qwen3_5_moe
