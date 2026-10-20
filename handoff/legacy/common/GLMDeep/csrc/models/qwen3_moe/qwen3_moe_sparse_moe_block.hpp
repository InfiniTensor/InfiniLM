#pragma once
#include "../../weights/weights.hpp"

#include "../../config/model_config.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "qwen3_moe_experts.hpp"
#include "qwen3_moe_topk_router.hpp"

#include <cstddef>
#include <memory>

namespace infinilm::models::qwen3_moe {

class Qwen3MoeSparseMoeBlock final : public infinilm::weights::ModelWeights {
public:
    Qwen3MoeSparseMoeBlock(infinilm::runtime::Runtime &, std::shared_ptr<config::ModelConfig>, size_t,
        const engine::distributed::RankInfo &, backends::AttentionBackend, const layers::moe::EPConfig &);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &hidden_states) const;

protected:
    std::shared_ptr<Qwen3MoeTopKRouter> gate_;
    std::shared_ptr<Qwen3MoeExperts> experts_;
    std::shared_ptr<infinilm::layers::moe::FusedMoE> fused_moe_;
};

} // namespace infinilm::models::qwen3_moe
