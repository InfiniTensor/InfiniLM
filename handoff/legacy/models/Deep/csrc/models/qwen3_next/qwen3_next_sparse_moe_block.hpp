#pragma once
#include "../../weights/weights.hpp"

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include <memory>

namespace infinilm::models::qwen3_next {

class Qwen3NextSharedExpert : public infinilm::weights::ModelWeights {
protected:
    infinilm::runtime::Device device_;

public:
    Qwen3NextSharedExpert(infinilm::runtime::Runtime &runtime,
                      std::shared_ptr<infinilm::config::ModelConfig> model_config,
                      const engine::distributed::RankInfo &rank_info,
                      backends::AttentionBackend attention_backend,
                      const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &hidden_states) const;
    void process_weights_after_loading(infinilm::runtime::Runtime &runtime, bool pre_transpose) override;

protected:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;
};

class Qwen3NextSparseMoeBlock : public infinilm::weights::ModelWeights {
public:
    Qwen3NextSparseMoeBlock(infinilm::runtime::Runtime &runtime,
                        std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                        const engine::distributed::RankInfo &rank_info,
                        backends::AttentionBackend attention_backend,
                        const layers::moe::EPConfig &ep_config);

    infinilm::runtime::Tensor forward(infinilm::runtime::Runtime &runtime,
                               const infinilm::runtime::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::moe::FusedMoeExperts> experts_;
    std::shared_ptr<infinilm::layers::moe::FusedMoE> fused_moe_;
    std::shared_ptr<Qwen3NextSharedExpert> shared_expert_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> shared_expert_gate_;
};

} // namespace infinilm::models::qwen3_next
