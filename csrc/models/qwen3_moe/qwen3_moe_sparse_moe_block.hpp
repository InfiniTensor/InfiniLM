#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"
#include "qwen3_moe_experts.hpp"
#include "qwen3_moe_topk_router.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::qwen3_moe {

class Qwen3MoeSparseMoeBlock final : public infinicore::nn::Module {
public:
    Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           const infinicore::Device &device);
    Qwen3MoeSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                           size_t layer_idx,
                           const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    std::shared_ptr<Qwen3MoeTopKRouter> legacy_gate_;
    std::shared_ptr<Qwen3MoeExperts> legacy_experts_;
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::moe::FusedMoeExperts> experts_;
    std::shared_ptr<infinilm::layers::moe::FusedMoE> fused_moe_;
    bool use_legacy_moe_{false};
};

} // namespace infinilm::models::qwen3_moe
