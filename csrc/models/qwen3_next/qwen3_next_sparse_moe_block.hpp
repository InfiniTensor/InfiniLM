#pragma once

#include "../../config/model_config.hpp"
#include "../../layers/linear/linear.hpp"
#include "../../layers/moe/experts/fused_moe_experts.hpp"
#include "../../layers/moe/fused_moe.hpp"
#include "../../layers/moe/router/topk_router.hpp"

#include <memory>

namespace infinilm::models::qwen3_next {

class Qwen3NextSharedExpert : public infinicore::nn::Module {
public:
    Qwen3NextSharedExpert(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                          const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;
    void process_weights_after_loading() override;
    void reset_runtime_state() const override;

protected:
    std::shared_ptr<infinilm::layers::linear::GateUpParallelLinear> gate_up_proj_;
    std::shared_ptr<infinilm::layers::linear::RowParallelLinear> down_proj_;
};

class Qwen3NextSparseMoeBlock : public infinicore::nn::Module {
public:
    Qwen3NextSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            const infinicore::Device &device);
    Qwen3NextSparseMoeBlock(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                            size_t layer_idx,
                            const infinicore::Device &device);

    infinicore::Tensor forward(const infinicore::Tensor &hidden_states) const;

protected:
    std::shared_ptr<infinilm::layers::moe::TopKRouter> gate_;
    std::shared_ptr<infinilm::layers::moe::FusedMoeExperts> experts_;
    std::shared_ptr<infinilm::layers::moe::FusedMoE> fused_moe_;
    std::shared_ptr<Qwen3NextSharedExpert> shared_expert_;
    std::shared_ptr<infinilm::layers::linear::ReplicatedLinear> shared_expert_gate_;
};

} // namespace infinilm::models::qwen3_next
